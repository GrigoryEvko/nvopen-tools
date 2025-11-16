// Function: sub_BC3A50
// Address: 0xbc3a50
//
__int64 __fastcall sub_BC3A50(__int64 a1, char a2, char a3)
{
  sub_C9E8E0(a1, "pass", 4, "Pass execution timing report", 28);
  sub_C9E8E0(a1 + 112, "analysis", 8, "Analysis execution timing report", 32);
  *(_BYTE *)(a1 + 416) = a2;
  *(_QWORD *)(a1 + 240) = 0x3800000000LL;
  *(_QWORD *)(a1 + 248) = a1 + 264;
  *(_BYTE *)(a1 + 417) = a3;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 256) = 0x800000000LL;
  *(_QWORD *)(a1 + 328) = a1 + 344;
  *(_QWORD *)(a1 + 336) = 0x800000000LL;
  *(_QWORD *)(a1 + 408) = 0;
  return 0x800000000LL;
}
