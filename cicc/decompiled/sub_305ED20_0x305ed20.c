// Function: sub_305ED20
// Address: 0x305ed20
//
char __fastcall sub_305ED20(__int64 a1)
{
  sub_E41FB0(*(_QWORD *)(a1 + 8), "GenericToNVVMPass]", 0x11u, "generic-to-nvvm", 0xFu);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "NVPTXCtorDtorLoweringPass]", 0x19u, "nvptx-lower-ctor-dtor", 0x15u);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "NVPTXSetGlobalArrayAlignmentPass]", 0x20u, "nvptx-set-global-array-alignment", 0x20u);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "NVVMReflectPass]", 0xFu, "nvvm-reflect", 0xCu);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "RegisterPressureAnalysis]", 0x18u, "register-pressure-analysis", 0x1Au);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "NVPTXAA]", 7u, "nvptx-aa", 8u);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "NVVMIntrRangePass]", 0x11u, "nvvm-intr-range", 0xFu);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "CodeGenPrepareSCEVPass]", 0x16u, "codegenpreparescev", 0x12u);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "LowerStructArgsPass]", 0x13u, "lower-struct-args", 0x11u);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "NVPTXSetLocalArrayAlignmentPass]", 0x1Fu, "nvptx-set-local-array-alignment", 0x1Fu);
  sub_E41FB0(*(_QWORD *)(a1 + 8), "NVPTXCopyByValArgsPass]", 0x16u, "nvptx-copy-byval-args", 0x15u);
  return sub_E41FB0(*(_QWORD *)(a1 + 8), "NVPTXLowerArgsPass]", 0x12u, "nvptx-lower-args", 0x10u);
}
