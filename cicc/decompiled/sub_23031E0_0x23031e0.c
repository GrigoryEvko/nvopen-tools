// Function: sub_23031E0
// Address: 0x23031e0
//
__int64 __fastcall sub_23031E0(__int64 a1, __int64 a2, __int64 **a3)
{
  __int64 v4; // rax
  unsigned int v5; // esi
  const char *v7; // [rsp+0h] [rbp-40h] BYREF
  char v8; // [rsp+20h] [rbp-20h]
  char v9; // [rsp+21h] [rbp-1Fh]

  v4 = sub_BCE3C0(*a3, 0);
  v9 = 1;
  v8 = 3;
  v5 = *(_DWORD *)(v4 + 8);
  v7 = "__bad_alias";
  sub_B30500((_QWORD *)v4, v5 >> 8, 7, (__int64)&v7, 0, (__int64)a3);
  memset((void *)a1, 0, 0x60u);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
