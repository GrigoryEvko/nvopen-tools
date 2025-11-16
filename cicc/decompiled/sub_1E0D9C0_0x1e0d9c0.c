// Function: sub_1E0D9C0
// Address: 0x1e0d9c0
//
__int64 __fastcall sub_1E0D9C0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v5; // rax
  _BYTE *v6; // rsi
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  char *v9; // rdi
  size_t v10; // r12
  __int64 v11; // rsi
  char *v13; // [rsp+0h] [rbp-40h] BYREF
  char *v14; // [rsp+8h] [rbp-38h]
  char *v15; // [rsp+10h] [rbp-30h]

  v5 = *(_BYTE **)(a2 + 8);
  v6 = *(_BYTE **)a2;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v7 = v5 - v6;
  if ( v5 == v6 )
  {
    v10 = 0;
    v9 = 0;
  }
  else
  {
    if ( v7 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, v6, a3);
    v8 = sub_22077B0(v7);
    v6 = *(_BYTE **)a2;
    v9 = (char *)v8;
    v5 = *(_BYTE **)(a2 + 8);
    v10 = (size_t)&v5[-*(_QWORD *)a2];
  }
  v13 = v9;
  v14 = v9;
  v15 = &v9[v7];
  if ( v5 != v6 )
    v9 = (char *)memmove(v9, v6, v10);
  v11 = *(_QWORD *)(a1 + 16);
  v14 = &v9[v10];
  if ( v11 == *(_QWORD *)(a1 + 24) )
  {
    sub_1E0D7A0((char **)(a1 + 8), (char *)v11, (__int64 *)&v13);
    v9 = v13;
    v7 = v15 - v13;
  }
  else
  {
    if ( v11 )
    {
      *(_QWORD *)v11 = v13;
      *(_QWORD *)(v11 + 8) = v14;
      *(_QWORD *)(v11 + 16) = v15;
      *(_QWORD *)(a1 + 16) += 24LL;
      return -1431655765 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3) - 1;
    }
    *(_QWORD *)(a1 + 16) = 24;
  }
  if ( v9 )
    j_j___libc_free_0(v9, v7);
  return -1431655765 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3) - 1;
}
