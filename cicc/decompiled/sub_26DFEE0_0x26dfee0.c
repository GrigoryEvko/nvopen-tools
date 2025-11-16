// Function: sub_26DFEE0
// Address: 0x26dfee0
//
_QWORD *__fastcall sub_26DFEE0(unsigned __int64 *a1, unsigned __int64 a2, unsigned __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v9; // rdi
  __int64 v10; // rsi
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  unsigned __int64 v14; // r13
  char *v15; // r15
  unsigned __int64 *v16; // r9
  _QWORD *v17; // rsi
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rcx
  unsigned __int64 v20; // rdx
  char *v21; // rax
  __int64 n; // [rsp+8h] [rbp-38h]

  v9 = (__int64)(a1 + 4);
  v10 = *(_QWORD *)(v9 - 24);
  if ( !(unsigned __int8)sub_222DA10(v9, v10, *(_QWORD *)(v9 - 8), a5) )
    goto LABEL_2;
  v14 = v11;
  if ( v11 == 1 )
  {
    v15 = (char *)(a1 + 6);
    a1[6] = 0;
    v16 = a1 + 6;
  }
  else
  {
    if ( v11 > 0xFFFFFFFFFFFFFFFLL )
      sub_4261EA(v9, v10, v11);
    n = 8 * v11;
    v15 = (char *)sub_22077B0(8 * v11);
    memset(v15, 0, n);
    v16 = a1 + 6;
  }
  v17 = (_QWORD *)a1[2];
  a1[2] = 0;
  if ( v17 )
  {
    v18 = 0;
    do
    {
      while ( 1 )
      {
        v19 = v17;
        v17 = (_QWORD *)*v17;
        v20 = v19[3] % v14;
        v21 = &v15[8 * v20];
        if ( !*(_QWORD *)v21 )
          break;
        *v19 = **(_QWORD **)v21;
        **(_QWORD **)v21 = v19;
LABEL_11:
        if ( !v17 )
          goto LABEL_15;
      }
      *v19 = a1[2];
      a1[2] = (unsigned __int64)v19;
      *(_QWORD *)v21 = a1 + 2;
      if ( !*v19 )
      {
        v18 = v20;
        goto LABEL_11;
      }
      *(_QWORD *)&v15[8 * v18] = v19;
      v18 = v20;
    }
    while ( v17 );
  }
LABEL_15:
  if ( (unsigned __int64 *)*a1 != v16 )
    j_j___libc_free_0(*a1);
  *a1 = (unsigned __int64)v15;
  a1[1] = v14;
  a2 = a3 % v14;
LABEL_2:
  a4[3] = a3;
  v12 = *(_QWORD **)(*a1 + 8 * a2);
  if ( v12 )
  {
    *a4 = *v12;
    **(_QWORD **)(*a1 + 8 * a2) = a4;
  }
  else
  {
    *a4 = a1[2];
    a1[2] = (unsigned __int64)a4;
    if ( *a4 )
      *(_QWORD *)(*a1 + 8 * (*(_QWORD *)(*a4 + 24LL) % a1[1])) = a4;
    *(_QWORD *)(*a1 + 8 * a2) = a1 + 2;
  }
  ++a1[3];
  return a4;
}
