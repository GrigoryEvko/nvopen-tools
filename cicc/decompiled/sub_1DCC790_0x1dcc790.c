// Function: sub_1DCC790
// Address: 0x1dcc790
//
char *__fastcall sub_1DCC790(char *a1, int a2)
{
  unsigned __int64 v2; // rax
  char *v3; // rdx
  char *v5; // rsi
  char *v6; // r15
  _QWORD *v7; // r13
  char *v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  char *v13; // rdi
  __int64 v14; // r13
  char *v15; // r13
  char *v16; // r15
  char *v17; // r13
  __int64 v18; // rdi
  char *v19; // r14
  char *v20; // rbx
  char *v21; // rdi
  char *v22; // [rsp+8h] [rbp-58h]
  unsigned int v23; // [rsp+14h] [rbp-4Ch]
  char *v24; // [rsp+18h] [rbp-48h]
  unsigned int v25; // [rsp+28h] [rbp-38h]
  int v26; // [rsp+2Ch] [rbp-34h]

  v2 = *((unsigned int *)a1 + 60);
  v24 = a1;
  v25 = a2 & 0x7FFFFFFF;
  v23 = (a2 & 0x7FFFFFFF) + 1;
  if ( v23 <= (unsigned int)v2 )
    goto LABEL_2;
  if ( v23 < v2 )
  {
    v3 = (char *)*((_QWORD *)a1 + 29);
    v15 = &v3[56 * v2];
    v16 = &v3[56 * v23];
    if ( v15 == v16 )
      goto LABEL_21;
    v17 = v15 - 48;
    do
    {
      v18 = *((_QWORD *)v17 + 3);
      v19 = v17 - 8;
      if ( v18 )
        j_j___libc_free_0(v18, *((_QWORD *)v17 + 5) - v18);
      v20 = *(char **)v17;
      while ( v20 != v17 )
      {
        v21 = v20;
        v20 = *(char **)v20;
        j_j___libc_free_0(v21, 40);
      }
      v17 -= 56;
    }
    while ( v16 != v19 );
LABEL_20:
    v3 = (char *)*((_QWORD *)v24 + 29);
LABEL_21:
    *((_DWORD *)v24 + 60) = v23;
    return &v3[56 * v25];
  }
  if ( v23 > v2 )
  {
    if ( v23 > (unsigned __int64)*((unsigned int *)a1 + 61) )
    {
      a1 += 232;
      sub_1DCC500((unsigned int *)v24 + 58, v23);
      v2 = *((unsigned int *)v24 + 60);
    }
    v3 = (char *)*((_QWORD *)v24 + 29);
    v5 = &v3[56 * v23];
    v22 = v5;
    v6 = &v3[56 * v2];
    if ( v5 == v6 )
      goto LABEL_21;
    do
    {
      if ( v6 )
      {
        v7 = v6 + 8;
        *(_QWORD *)v6 = 0;
        *((_QWORD *)v6 + 2) = v6 + 8;
        *((_QWORD *)v6 + 1) = v6 + 8;
        *((_QWORD *)v6 + 3) = 0;
        v8 = (char *)*((_QWORD *)v24 + 32);
        if ( v8 != v24 + 256 )
        {
          do
          {
            v9 = *((_QWORD *)v8 + 3);
            v10 = *((_QWORD *)v8 + 4);
            v26 = *((_DWORD *)v8 + 4);
            v11 = sub_22077B0(40);
            v5 = v6 + 8;
            *(_QWORD *)(v11 + 24) = v9;
            a1 = (char *)v11;
            *(_DWORD *)(v11 + 16) = v26;
            *(_QWORD *)(v11 + 32) = v10;
            sub_2208C80(v11, v6 + 8);
            ++*((_QWORD *)v6 + 3);
            v8 = *(char **)v8;
          }
          while ( v8 != v24 + 256 );
          v7 = (_QWORD *)*((_QWORD *)v6 + 1);
        }
        *(_QWORD *)v6 = v7;
        v12 = *((_QWORD *)v24 + 36) - *((_QWORD *)v24 + 35);
        *((_QWORD *)v6 + 4) = 0;
        *((_QWORD *)v6 + 5) = 0;
        *((_QWORD *)v6 + 6) = 0;
        if ( v12 )
        {
          if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
            sub_4261EA(a1, v5, v3);
          v13 = (char *)sub_22077B0(v12);
        }
        else
        {
          v12 = 0;
          v13 = 0;
        }
        v3 = v24;
        *((_QWORD *)v6 + 4) = v13;
        *((_QWORD *)v6 + 6) = &v13[v12];
        *((_QWORD *)v6 + 5) = v13;
        v5 = (char *)*((_QWORD *)v24 + 35);
        v14 = *((_QWORD *)v24 + 36) - (_QWORD)v5;
        if ( *((char **)v24 + 36) != v5 )
          v13 = (char *)memmove(v13, v5, *((_QWORD *)v24 + 36) - (_QWORD)v5);
        a1 = &v13[v14];
        *((_QWORD *)v6 + 5) = a1;
      }
      v6 += 56;
    }
    while ( v22 != v6 );
    goto LABEL_20;
  }
LABEL_2:
  v3 = (char *)*((_QWORD *)a1 + 29);
  return &v3[56 * v25];
}
