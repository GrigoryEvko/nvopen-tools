// Function: sub_22D0390
// Address: 0x22d0390
//
__int64 __fastcall sub_22D0390(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9

  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_22CFE90(a1, (__int64)&unk_4F81450, a3, a4, a5, a6);
  sub_22CFE90(a1, (__int64)&unk_4F875F0, v6, v7, v8, v9);
  sub_22CFE90(a1, (__int64)&unk_4FDBCE0, v10, v11, v12, v13);
  sub_22CFE90(a1, (__int64)&unk_4F881D0, v14, v15, v16, v17);
  return a1;
}
