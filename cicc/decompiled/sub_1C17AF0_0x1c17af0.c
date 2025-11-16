// Function: sub_1C17AF0
// Address: 0x1c17af0
//
__int64 __fastcall sub_1C17AF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // rdx
  __int64 v11; // r12
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // r14
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi

  v8 = sub_22077B0(152);
  v9 = v8;
  if ( v8 )
    sub_1C17480(v8, a2, a3, a4, a5, 0, 0);
  *(_QWORD *)a1 = v9;
  v10 = (_QWORD *)sub_22077B0(104);
  if ( v10 )
  {
    memset(v10, 0, 0x68u);
    v10[11] = 1;
    v10[2] = v10 + 4;
    v10[3] = 0x400000000LL;
    v10[8] = v10 + 10;
  }
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 144LL);
  *(_QWORD *)(*(_QWORD *)a1 + 144LL) = v10;
  if ( v11 )
  {
    v12 = *(unsigned __int64 **)(v11 + 16);
    v13 = &v12[*(unsigned int *)(v11 + 24)];
    while ( v13 != v12 )
    {
      v14 = *v12++;
      _libc_free(v14);
    }
    v15 = *(unsigned __int64 **)(v11 + 64);
    v16 = (unsigned __int64)&v15[2 * *(unsigned int *)(v11 + 72)];
    if ( v15 != (unsigned __int64 *)v16 )
    {
      do
      {
        v17 = *v15;
        v15 += 2;
        _libc_free(v17);
      }
      while ( (unsigned __int64 *)v16 != v15 );
      v16 = *(_QWORD *)(v11 + 64);
    }
    if ( v16 != v11 + 80 )
      _libc_free(v16);
    v18 = *(_QWORD *)(v11 + 16);
    if ( v18 != v11 + 32 )
      _libc_free(v18);
    j_j___libc_free_0(v11, 104);
  }
  return a1;
}
