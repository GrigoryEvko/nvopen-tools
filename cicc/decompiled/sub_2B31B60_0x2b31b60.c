// Function: sub_2B31B60
// Address: 0x2b31b60
//
_QWORD *__fastcall sub_2B31B60(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r13
  __int64 v9; // rdi
  unsigned int v11; // esi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  char *v16; // [rsp+0h] [rbp-70h] BYREF
  __int64 v17; // [rsp+8h] [rbp-68h]
  _BYTE v18[96]; // [rsp+10h] [rbp-60h] BYREF

  v6 = a1 + 2;
  if ( *(_DWORD *)(a2 + 104) == 5 )
  {
    *a1 = v6;
    a1[1] = 0xC00000000LL;
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 144);
    v11 = *(_DWORD *)(a2 + 152);
    v17 = 0xC00000000LL;
    v16 = v18;
    sub_2B0FC00(v9, v11, (__int64)&v16, a4, a5, a6);
    sub_2B319A0((__int64)&v16, *(int **)(a2 + 112), *(unsigned int *)(a2 + 120));
    *a1 = v6;
    a1[1] = 0xC00000000LL;
    if ( (_DWORD)v17 )
      sub_2B0D090((__int64)a1, &v16, v12, v13, v14, v15);
    if ( v16 != v18 )
      _libc_free((unsigned __int64)v16);
  }
  return a1;
}
