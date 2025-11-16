// Function: sub_2DD41F0
// Address: 0x2dd41f0
//
__int64 __fastcall sub_2DD41F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 *v9; // r15
  int v10; // ebx
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 *v14; // r15
  int v15; // ebx
  __int64 v16; // rdi
  char **v17; // rsi
  __int64 v18; // r15
  unsigned __int64 v19; // rbx
  __int64 v20; // r14
  char **v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r14
  unsigned __int64 v25; // r13
  __int64 v26; // rbx
  char **v27; // rsi
  __int64 v28; // rdi
  char **v30; // rsi
  unsigned int v33; // [rsp+14h] [rbp-3Ch]
  __int64 v34; // [rsp+18h] [rbp-38h]

  v6 = a1;
  v7 = a3;
  v34 = a5;
  if ( a3 != a4 && a1 != a2 )
  {
    do
    {
      v8 = *(_QWORD *)v7 + 8LL * *(unsigned int *)(v7 + 8);
      if ( *(_QWORD *)v7 == v8 )
      {
        v10 = 0;
      }
      else
      {
        v9 = *(__int64 **)v7;
        v10 = 0;
        do
        {
          v11 = *v9++;
          v10 += sub_39FAC40(v11);
        }
        while ( (__int64 *)v8 != v9 );
      }
      v12 = *(_QWORD *)v6;
      v13 = *(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8);
      v33 = *(_DWORD *)(v7 + 72) * v10;
      if ( *(_QWORD *)v6 == v13 )
        goto LABEL_26;
      v14 = *(__int64 **)v6;
      v15 = 0;
      do
      {
        v16 = *v14++;
        v15 += sub_39FAC40(v16);
      }
      while ( (__int64 *)v13 != v14 );
      if ( v33 >= *(_DWORD *)(v6 + 72) * v15 )
      {
LABEL_26:
        v30 = (char **)v6;
        v6 += 80;
        sub_2DD3500(v34, v30, v12, a4, a5, a6);
        *(_DWORD *)(v34 + 64) = *(_DWORD *)(v6 - 16);
        *(_DWORD *)(v34 + 72) = *(_DWORD *)(v6 - 8);
      }
      else
      {
        v17 = (char **)v7;
        v7 += 80;
        sub_2DD3500(v34, v17, v12, a4, a5, a6);
        *(_DWORD *)(v34 + 64) = *(_DWORD *)(v7 - 16);
        *(_DWORD *)(v34 + 72) = *(_DWORD *)(v7 - 8);
      }
      v34 += 80;
    }
    while ( v6 != a2 && v7 != a4 );
  }
  v18 = a2 - v6;
  v19 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - v6) >> 4);
  if ( a2 - v6 > 0 )
  {
    v20 = v34;
    do
    {
      v21 = (char **)v6;
      v22 = v20;
      v6 += 80;
      v20 += 80;
      sub_2DD3500(v22, v21, a3, a4, a5, a6);
      *(_DWORD *)(v20 - 16) = *(_DWORD *)(v6 - 16);
      a3 = *(unsigned int *)(v6 - 8);
      *(_DWORD *)(v20 - 8) = a3;
      --v19;
    }
    while ( v19 );
    v23 = 80;
    if ( v18 > 0 )
      v23 = v18;
    v34 += v23;
  }
  v24 = a4 - v7;
  v25 = 0xCCCCCCCCCCCCCCCDLL * ((a4 - v7) >> 4);
  if ( a4 - v7 > 0 )
  {
    v26 = v34;
    do
    {
      v27 = (char **)v7;
      v28 = v26;
      v7 += 80;
      v26 += 80;
      sub_2DD3500(v28, v27, a3, a4, a5, a6);
      *(_DWORD *)(v26 - 16) = *(_DWORD *)(v7 - 16);
      *(_DWORD *)(v26 - 8) = *(_DWORD *)(v7 - 8);
      --v25;
    }
    while ( v25 );
    if ( v24 <= 0 )
      v24 = 80;
    v34 += v24;
  }
  return v34;
}
