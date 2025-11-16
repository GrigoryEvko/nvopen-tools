// Function: sub_B4FC50
// Address: 0xb4fc50
//
__int64 __fastcall sub_B4FC50(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rax
  int v4; // ecx
  __int64 v6; // rsi
  __int64 v7; // rdi
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  _BYTE v10[48]; // [rsp+10h] [rbp-30h] BYREF

  v2 = 0;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL);
  if ( *(_BYTE *)(v3 + 8) == 17 )
  {
    v4 = *(_DWORD *)(v3 + 32);
    v6 = *(unsigned int *)(a1 + 80);
    v7 = *(_QWORD *)(a1 + 72);
    v9[0] = v10;
    v9[1] = 0x800000000LL;
    v2 = sub_B4FA40(v7, v6, a2, 2 * v4, (__int64)v9);
    if ( (_BYTE *)v9[0] != v10 )
      _libc_free(v9[0], v6);
  }
  return v2;
}
