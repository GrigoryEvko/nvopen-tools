// Function: sub_22ADC50
// Address: 0x22adc50
//
char *__fastcall sub_22ADC50(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v7; // r12
  unsigned int *v8; // rbx
  char *v10; // r13
  char **v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdi
  char **v14; // rsi
  __int64 v15; // r14
  unsigned __int64 v16; // r13
  _DWORD *v17; // r15
  char **v18; // rbx
  int v19; // eax
  char **v20; // rsi
  __int64 v21; // rdi
  __int64 v23; // r15
  unsigned __int64 v24; // r14
  _DWORD *v25; // rbx
  char **v26; // r13
  __int64 v27; // rdx
  char **v28; // rsi
  __int64 v29; // rdi

  v7 = (_DWORD *)a5;
  v8 = (unsigned int *)a3;
  if ( a1 == a2 )
  {
LABEL_7:
    v15 = a4 - (_QWORD)v8;
    v16 = 0x8E38E38E38E38E39LL * (v15 >> 3);
    if ( v15 > 0 )
    {
      v17 = v7 + 2;
      v18 = (char **)(v8 + 2);
      do
      {
        v19 = *((_DWORD *)v18 - 2);
        v20 = v18;
        v21 = (__int64)v17;
        v18 += 9;
        v17 += 18;
        *(v17 - 20) = v19;
        sub_22AD4A0(v21, v20, a3, a4, a5, a6);
        --v16;
      }
      while ( v16 );
      return (char *)v7 + v15;
    }
    return (char *)v7;
  }
  v10 = a1;
  while ( (unsigned int *)a4 != v8 )
  {
    v12 = *v8;
    v13 = (__int64)(v7 + 2);
    if ( (unsigned int)v12 > *(_DWORD *)v10 )
    {
      *v7 = v12;
      v11 = (char **)(v8 + 2);
      v7 += 18;
      v8 += 18;
      sub_22AD4A0(v13, v11, v12, a4, a5, a6);
      if ( v10 == a2 )
        goto LABEL_7;
    }
    else
    {
      *v7 = *(_DWORD *)v10;
      v14 = (char **)(v10 + 8);
      v10 += 72;
      v7 += 18;
      sub_22AD4A0(v13, v14, v12, a4, a5, a6);
      if ( v10 == a2 )
        goto LABEL_7;
    }
  }
  v23 = a2 - v10;
  v24 = 0x8E38E38E38E38E39LL * (v23 >> 3);
  if ( v23 <= 0 )
    return (char *)v7;
  v25 = v7 + 2;
  v26 = (char **)(v10 + 8);
  do
  {
    v27 = *((unsigned int *)v26 - 2);
    v28 = v26;
    v29 = (__int64)v25;
    v26 += 9;
    v25 += 18;
    *(v25 - 20) = v27;
    sub_22AD4A0(v29, v28, v27, a4, a5, a6);
    --v24;
  }
  while ( v24 );
  return (char *)v7 + v23;
}
