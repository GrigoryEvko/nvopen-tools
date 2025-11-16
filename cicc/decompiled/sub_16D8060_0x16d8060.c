// Function: sub_16D8060
// Address: 0x16d8060
//
char __fastcall sub_16D8060(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9

  sub_2241130(a1 + 64, 0, *(_QWORD *)(a1 + 72), a2, a3);
  sub_2241130(a1 + 96, 0, *(_QWORD *)(a1 + 104), a4, a5);
  *(_QWORD *)(a1 + 136) = a6;
  *(_WORD *)(a1 + 128) = 0;
  return sub_16D7F90(a6, a1, v9, v10, v11, v12);
}
