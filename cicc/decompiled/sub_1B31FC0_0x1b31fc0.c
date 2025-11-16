// Function: sub_1B31FC0
// Address: 0x1b31fc0
//
__int64 __fastcall sub_1B31FC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 **a5)
{
  _QWORD *v6; // rbx
  const void *v7; // rax
  const void *v8; // rsi
  unsigned __int64 v9; // rdx
  __int64 v11; // rax
  __int64 *v12; // r12
  __int64 *v13; // r8
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 *v16; // r13
  __int64 *v17; // r15
  __int64 *v18; // r14
  __int64 v19; // rsi
  __int64 result; // rax
  __int64 *v21; // rbx
  __int64 v24; // [rsp+10h] [rbp-60h]
  size_t n; // [rsp+20h] [rbp-50h]
  unsigned __int64 v26; // [rsp+28h] [rbp-48h]
  unsigned __int64 v27; // [rsp+30h] [rbp-40h]
  char *dest; // [rsp+38h] [rbp-38h]

  v6 = a1;
  v7 = *(const void **)(a4 + 8);
  v8 = *(const void **)a4;
  v9 = (unsigned __int64)v7 - *(_QWORD *)a4;
  v27 = v9;
  if ( v7 == *(const void **)a4 )
  {
    n = 0;
    dest = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_30;
    v11 = sub_22077B0(v9);
    v8 = *(const void **)a4;
    dest = (char *)v11;
    v7 = *(const void **)(a4 + 8);
    v9 = (unsigned __int64)v7 - *(_QWORD *)a4;
    n = v9;
  }
  a1 = dest;
  if ( v8 != v7 )
    memmove(dest, v8, n);
  v12 = a5[1];
  v13 = *a5;
  v14 = (char *)v12 - (char *)*a5;
  v26 = v14;
  if ( !v14 )
  {
    v16 = 0;
    v24 = 0;
    if ( v13 != v12 )
      goto LABEL_9;
LABEL_27:
    if ( v6 )
    {
      v18 = v16;
LABEL_15:
      v6[5] = v16;
      v6[6] = v18;
      *v6 = a2;
      v6[1] = a3;
      v6[2] = dest;
      v6[3] = &dest[n];
      v6[4] = &dest[v27];
      result = v24;
      v6[7] = v24;
      return result;
    }
LABEL_22:
    if ( !v16 )
      goto LABEL_24;
    goto LABEL_23;
  }
  if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_30:
    sub_4261EA(a1, v8, v9);
  v15 = sub_22077B0(v14);
  v12 = a5[1];
  v13 = *a5;
  v16 = (__int64 *)v15;
  v24 = v15 + v26;
  if ( *a5 == v12 )
    goto LABEL_27;
LABEL_9:
  v17 = v13;
  v18 = v16;
  do
  {
    if ( v18 )
    {
      v19 = *v17;
      *v18 = *v17;
      if ( v19 )
        sub_1623A60((__int64)v18, v19, 2);
    }
    ++v17;
    ++v18;
  }
  while ( v12 != v17 );
  if ( v6 )
    goto LABEL_15;
  if ( v18 != v16 )
  {
    v21 = v16;
    do
    {
      if ( *v21 )
        sub_161E7C0((__int64)v21, *v21);
      ++v21;
    }
    while ( v18 != v21 );
    goto LABEL_22;
  }
LABEL_23:
  j_j___libc_free_0(v16, v26);
LABEL_24:
  result = (__int64)dest;
  if ( dest )
    return j_j___libc_free_0(dest, v27);
  return result;
}
