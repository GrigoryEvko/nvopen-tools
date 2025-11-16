// Function: sub_A55860
// Address: 0xa55860
//
__int64 __fastcall sub_A55860(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)a1 = &unk_49D9A00;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = a2;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  return 0;
}
