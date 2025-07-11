from manim import *
import numpy as np
import random
from itertools import product
import optuna

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern



config.quality = "low_quality"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30




class SearchSpace3D(ThreeDScene):
    def construct(self):
        # --- 2D Table Scene ---
        headline2d = Text("Hyperparameter Optimization using Optuna")
        headline2d.to_edge(UP)

        subtitle2d = Text("A practical example: optimize LR and LR-Decay in a vision transformer", font_size=24).next_to(headline2d, DOWN, buff=0.5)

        # Parameter-Tabelle
        data = [
            ["Parameter", "Range / Options", "Type"],
            ["Learning Rate", "10^{-6} – 10^{-4} (log)", "Continuous"],
            ["Layer-wise LR Decay", "0.6 – 1.0", "Continuous"],
        ]
        table2d = Table(
            data,
            include_outer_lines=True,
            line_config={"stroke_width": 2}
        ).scale(0.3)
        table2d.next_to(subtitle2d, DOWN, buff=0.5)

        # Brace + Theta-Label
        theta_brace = Brace(table2d, direction=LEFT, buff=0.1)
        theta_label = Tex(r"$\Theta$").scale(0.6)
        theta_brace.put_at_tip(theta_label)

        # Goal-Label
        opt_label = Tex(r"Objective:\ Find\ $\theta$\ such\ that:").scale(0.6)
        opt_label.next_to(table2d,DOWN, buff=1)

        # Argmin-Ausdruck
        argmin_expr = MathTex(
            r"\theta^* = \arg\min_{\theta \in \Theta} -F_{1}\bigl(y,\hat y(\theta)\bigr)"
        ).scale(0.6)
        argmin_expr.next_to(opt_label, DOWN, buff=0.5)


        self.play(
            FadeIn(headline2d, shift=DOWN * 0.5), FadeIn(subtitle2d, shift=DOWN),
            Create(table2d),
        )
        self.play(
            Create(theta_brace),
            Write(theta_label),
        )
        self.play(
            Write(opt_label, run_time=0.5),
            Write(argmin_expr, run_time=1),
        )
        self.wait(2)

        self.play(
            FadeOut(headline2d), FadeOut(table2d), FadeOut(theta_brace), FadeOut(theta_label),
            FadeOut(opt_label),
            FadeOut(argmin_expr),
            FadeOut(subtitle2d),
            run_time=1
        )




        # --- 3D Search-Space Scene ---
        headline3d = Text("Hyperparameter Space")
        headline3d.to_edge(UP)
        self.add_fixed_in_frame_mobjects(headline3d)
        self.play(FadeIn(headline3d, shift=DOWN * 0.5))

        axes = ThreeDAxes(
            x_range=[-10, 10, 1], y_range=[-10, 10, 1], z_range=[0, 1, 0.1],
            x_length=8, y_length=8, z_length=5
        ).center()
        x_label, y_label, z_label = axes.get_axis_labels(
            x_label="Layer-wise LR Decay",
            y_label="learning rate",
            z_label="-F_1"
        ).submobjects
        x_label.rotate(PI / 2, axis=RIGHT)
        y_label.rotate(PI / 2, axis=UP)

        self.set_camera_orientation(phi=80 * DEGREES, theta=-40 * DEGREES, distance=10)
        self.play(Create(axes), Create(x_label), Create(y_label), Create(z_label))
        self.wait(1)

        self.begin_ambient_camera_rotation(-0.1)

        # Definition of function
        def param_fn(u, v):
            x, y = u, v
            g1 = 10.0 * np.exp(-((x + 2)**2 + (y + 2)**2) / 1.0)
            g2 = 14.5 * np.exp(-((x - 0)**2 + (y - 3)**2) / 2.0)
            g3 = 5.0 * np.exp(-((x - 3)**2 + (y + 1)**2) / 1.5)
            z = 1 - (g1 + g2 + g3) / (2.0 + 1.5 + 1.0)
            return np.array([x, y, z])

        surface = Surface(
            param_fn,
            u_range=[-5, 5], v_range=[-5, 5],
            resolution=(64, 64),
            checkerboard_colors=[TEAL_D, TEAL_E],
            fill_opacity=0.5,
        )
        self.play(Create(surface), run_time=3)
        self.wait(2)



        # --- Hide axes and labels, keep only surface ---
        self.play(
            FadeOut(axes), FadeOut(x_label), FadeOut(y_label), FadeOut(z_label),
            FadeOut(headline3d),
            run_time=1
        )

        # --- Camera move upward focusing on surface (use move_camera directly) ---
        self.move_camera(phi=60 * DEGREES, theta=-40 * DEGREES, distance=10, run_time=2)
        self.stop_ambient_camera_rotation()








        # --- Simulate Grid Search Points using Dot3D ---
        headline2d = Text("Grid-Search")
        headline2d.to_edge(UP)
        self.add_fixed_in_frame_mobjects(headline2d)
        self.play(FadeIn(headline2d), run_time=1 )

        self.begin_ambient_camera_rotation(0.1)
        sample_u = np.linspace(-5, 5, 6)
        sample_v = np.linspace(-5, 5, 6)
        samples = [param_fn(u, v) for u in sample_u for v in sample_v]
        dots = VGroup(*[
            Dot3D(point=pt, radius=0.05)
            for pt in samples
        ])
        self.play(LaggedStartMap(FadeIn, dots, shift=UP * 0.2), run_time=2)
        self.wait(1)

        # Determine best point index (minimal loss => minimal z)
        best_idx = int(np.argmin([pt[2] for pt in samples]))

        # --- Switch to top-down view ---
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=90 * DEGREES, theta=-90 * DEGREES, distance=8, run_time=2)
        self.wait(0.5)

        # --- Highlight best point from above ---
        # Create a ring on the surface plane
        ring = Circle(radius=0.3).rotate(PI/2, axis=RIGHT).move_to(samples[best_idx])
        ring.set_stroke(color=YELLOW, width=4).set_fill(opacity=0)

        label_grid = Tex("Best Grid Trial").scale(0.5)
        # Position label at 3D location but always face camera
        label_grid.move_to(samples[best_idx] + np.array([0, 0, 1]))
        self.add(label_grid)
        self.add_fixed_orientation_mobjects(label_grid)

        self.play(Create(ring), Write(label_grid), dots[best_idx].animate.set_color(YELLOW).scale(1.5), run_time=1)
        self.wait(2)


        self.move_camera(phi=60 * DEGREES, theta=-40 * DEGREES, distance=10, run_time=2)
        self.wait(2)

        self.play(
            *[FadeOut(dot) for dot in dots],
            FadeOut(headline2d), 
            run_time=1
        )
        self.wait(0.5)



        
        
        
        
        # Optuna-Study
        headline2d = Text("Optuna bayesian optimizaion")
        headline2d.to_edge(UP)
        self.add_fixed_in_frame_mobjects(headline2d)
        self.play(FadeIn(headline2d), run_time=1 )

        study = optuna.create_study(direction="minimize")
        def objective(trial):
            u = trial.suggest_uniform("u", -5, 5)
            v = trial.suggest_uniform("v", -5, 5)
            return param_fn(u, v)[2]
        study.optimize(objective, n_trials=len(samples)) # same amount of samples as gridsearch

        opt_pts = [
            param_fn(t.params["u"], t.params["v"])
            for t in study.trials
        ]
        dots_optuna = VGroup(*[
            Dot3D(point=pt, radius=0.05)
            for pt in opt_pts
        ])

        self.play(
            LaggedStartMap(FadeIn, dots_optuna, shift=UP * 0.2),
            run_time=5
        )
        self.wait(1)

        self.stop_ambient_camera_rotation()
        self.move_camera(phi=90 * DEGREES, theta=-90 * DEGREES, distance=8, run_time=2)
        self.wait(0.5)

        best_idx_opt = int(np.argmin([pt[2] for pt in opt_pts]))
        ring_opt = Circle(radius=0.3).rotate(PI/2, axis=RIGHT).move_to(opt_pts[best_idx_opt])
        ring_opt.set_stroke(color=RED, width=4).set_fill(opacity=0)
        
        label_opt = Tex("Best Optuna Trial").scale(0.5)
        label_opt.move_to(opt_pts[best_idx_opt] + np.array([0, 0, 1]))
        self.add(label_opt)
        self.add_fixed_orientation_mobjects(label_opt)

        self.play(Create(ring_opt), Write(label_opt), dots_optuna[best_idx_opt].animate.set_color(RED).scale(1.5), run_time=1)
        self.wait(2)

        self.move_camera(phi=60 * DEGREES, theta=-40 * DEGREES, distance=10, run_time=2)
        self.wait(1)

