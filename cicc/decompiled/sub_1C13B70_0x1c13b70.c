// Function: sub_1C13B70
// Address: 0x1c13b70
//
void __fastcall sub_1C13B70(__int64 a1, char **a2, __int64 a3)
{
  char *v4; // r14
  char *v5; // r12
  char *v6; // rdi
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // r8
  char *v9; // rax
  size_t v10; // rdx
  char *v11; // r15
  __int64 v12; // rax
  _QWORD *v13; // rbx
  char *v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // r14
  char *v17; // rdx
  char *v18; // r14

  if ( a2 != (char **)a1 )
  {
    v4 = a2[1];
    v5 = *a2;
    v6 = *(char **)a1;
    v7 = v4 - *a2;
    v8 = *(_QWORD *)(a1 + 16) - (_QWORD)v6;
    if ( v7 > v8 )
    {
      if ( v7 )
      {
        if ( v7 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(v6, a2, a3);
        v12 = sub_22077B0(a2[1] - *a2);
        v6 = *(char **)a1;
        v13 = (_QWORD *)v12;
        v8 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
      }
      else
      {
        v13 = 0;
      }
      if ( v4 != v5 )
      {
        v14 = v5;
        v15 = v13;
        v16 = (_QWORD *)((char *)v13 + v4 - v5);
        do
        {
          if ( v15 )
            *v15 = *(_QWORD *)v14;
          ++v15;
          v14 += 8;
        }
        while ( v16 != v15 );
      }
      if ( v6 )
        j_j___libc_free_0(v6, v8);
      v11 = (char *)v13 + v7;
      *(_QWORD *)a1 = v13;
      *(_QWORD *)(a1 + 16) = v11;
      goto LABEL_7;
    }
    v9 = *(char **)(a1 + 8);
    v10 = v9 - v6;
    if ( v7 > v9 - v6 )
    {
      if ( v10 )
      {
        memmove(v6, *a2, v10);
        v9 = *(char **)(a1 + 8);
        v6 = *(char **)a1;
        v4 = a2[1];
        v5 = *a2;
        v10 = (size_t)&v9[-*(_QWORD *)a1];
      }
      v17 = &v5[v10];
      if ( v17 != v4 )
      {
        v18 = &v9[v4 - v17];
        do
        {
          if ( v9 )
            *(_QWORD *)v9 = *(_QWORD *)v17;
          v9 += 8;
          v17 += 8;
        }
        while ( v18 != v9 );
        v11 = (char *)(*(_QWORD *)a1 + v7);
        goto LABEL_7;
      }
    }
    else if ( v4 != v5 )
    {
      memmove(v6, *a2, a2[1] - *a2);
      v6 = *(char **)a1;
    }
    v11 = &v6[v7];
LABEL_7:
    *(_QWORD *)(a1 + 8) = v11;
  }
}
