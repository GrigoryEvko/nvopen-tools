// Function: sub_D50BF0
// Address: 0xd50bf0
//
__int64 __fastcall sub_D50BF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx

  v6 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  sub_BC1CD0(a4, &unk_4F81450, a3);
  sub_D50AF0(v6 + 8);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&unk_4F82400);
  return a1;
}
