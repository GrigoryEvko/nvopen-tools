// Function: sub_C149C0
// Address: 0xc149c0
//
__int64 __fastcall sub_C149C0(__int64 a1, __int64 a2, __int64 a3)
{
  sub_E98A20();
  *(_QWORD *)(a1 + 296) = a3;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)a1 = &unk_49DB460;
  *(_QWORD *)(a1 + 320) = 0x1000000000LL;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_DWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = a1 + 376;
  *(_QWORD *)(a1 + 368) = 0;
  return a1 + 376;
}
