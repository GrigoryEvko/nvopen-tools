// Function: sub_7702C0
// Address: 0x7702c0
//
_QWORD *__fastcall sub_7702C0(__int64 a1)
{
  unsigned int v2; // r15d
  int *v3; // r13
  unsigned int v4; // r8d
  int v5; // eax
  int v6; // edx
  int v7; // esi
  int v8; // esi
  __int64 v9; // rax
  unsigned __int64 v10; // r14
  size_t v11; // rdx
  _QWORD *v12; // rdi
  int v13; // esi
  char *i; // rdx
  int *v15; // rcx
  char *v16; // rdi
  __int64 v17; // r9
  int v18; // r8d
  unsigned int v19; // eax
  __int64 v21; // rax
  __int64 v22; // rax
  size_t v23; // [rsp+8h] [rbp-38h]
  size_t v24; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(int **)a1;
  v4 = 8 * (v2 + 1);
  if ( v2 )
  {
    v5 = *(_DWORD *)(a1 + 8);
    v6 = 0;
    do
    {
      v7 = v6++;
      v5 &= v5 - 1;
    }
    while ( v5 );
    v8 = v7 + 2;
    v9 = v8;
    v10 = v8 - 1LL;
    if ( v8 > 10 )
    {
      v23 = v4;
      v21 = sub_822B10(v4);
      v11 = v23;
      v12 = (_QWORD *)v21;
      goto LABEL_7;
    }
  }
  else
  {
    v10 = 0;
    v9 = 1;
  }
  v11 = v4;
  v12 = (_QWORD *)qword_4F082C0[v9];
  if ( v12 )
  {
    qword_4F082C0[v9] = *v12;
  }
  else
  {
    v24 = v4;
    v22 = sub_823970(v4);
    v11 = v24;
    v12 = (_QWORD *)v22;
  }
LABEL_7:
  v13 = 2 * v2 + 1;
  v16 = (char *)memset(v12, 0, v11);
  if ( v2 != -1 )
  {
    v15 = v3;
    v17 = (__int64)&v3[v2 + 1];
    do
    {
      while ( 1 )
      {
        v18 = *v15;
        if ( *v15 )
          break;
        if ( (int *)v17 == ++v15 )
          goto LABEL_14;
      }
      v19 = v18 & v13;
      for ( i = &v16[4 * (v18 & v13)]; *(_DWORD *)i; i = &v16[4 * v19] )
        v19 = v13 & (v19 + 1);
      ++v15;
      *(_DWORD *)i = v18;
    }
    while ( (int *)v17 != v15 );
  }
LABEL_14:
  *(_QWORD *)a1 = v16;
  *(_DWORD *)(a1 + 8) = v13;
  if ( v10 > 0xA )
    return (_QWORD *)sub_822B90(v3, 4 * (v2 + 1), i, v15);
  *(_QWORD *)v3 = qword_4F082C0[v10];
  qword_4F082C0[v10] = v3;
  return qword_4F082C0;
}
