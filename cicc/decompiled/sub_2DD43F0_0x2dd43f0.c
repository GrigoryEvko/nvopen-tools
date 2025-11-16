// Function: sub_2DD43F0
// Address: 0x2dd43f0
//
__int64 __fastcall sub_2DD43F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r13
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
  __int64 v18; // r14
  unsigned __int64 v19; // r13
  __int64 v20; // rbx
  char **v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rax
  char **v25; // rsi
  __int64 v26; // r15
  unsigned __int64 v27; // rbx
  __int64 v28; // r14
  char **v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rax
  unsigned int v34; // [rsp+14h] [rbp-3Ch]
  __int64 v35; // [rsp+18h] [rbp-38h]

  v6 = a3;
  v35 = a5;
  if ( a2 != a1 )
  {
    v7 = a1;
    while ( a4 != v6 )
    {
      v8 = *(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8);
      if ( *(_QWORD *)v6 == v8 )
      {
        v10 = 0;
      }
      else
      {
        v9 = *(__int64 **)v6;
        v10 = 0;
        do
        {
          v11 = *v9++;
          v10 += sub_39FAC40(v11);
        }
        while ( (__int64 *)v8 != v9 );
      }
      v12 = *(_QWORD *)v7;
      v13 = *(_QWORD *)v7 + 8LL * *(unsigned int *)(v7 + 8);
      v34 = *(_DWORD *)(v6 + 72) * v10;
      if ( *(_QWORD *)v7 == v13 )
        goto LABEL_19;
      v14 = *(__int64 **)v7;
      v15 = 0;
      do
      {
        v16 = *v14++;
        v15 += sub_39FAC40(v16);
      }
      while ( v14 != (__int64 *)v13 );
      if ( v34 >= *(_DWORD *)(v7 + 72) * v15 )
      {
LABEL_19:
        v25 = (char **)v7;
        v7 += 80;
        sub_2DD3500(v35, v25, v12, a4, a5, a6);
        *(_DWORD *)(v35 + 64) = *(_DWORD *)(v7 - 16);
        *(_DWORD *)(v35 + 72) = *(_DWORD *)(v7 - 8);
      }
      else
      {
        v17 = (char **)v6;
        v6 += 80;
        sub_2DD3500(v35, v17, v12, a4, a5, a6);
        *(_DWORD *)(v35 + 64) = *(_DWORD *)(v6 - 16);
        *(_DWORD *)(v35 + 72) = *(_DWORD *)(v6 - 8);
      }
      v35 += 80;
      if ( a2 == v7 )
        goto LABEL_13;
    }
    v26 = a2 - v7;
    v27 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - v7) >> 4);
    if ( a2 - v7 <= 0 )
      return v35;
    v28 = v35;
    do
    {
      v29 = (char **)v7;
      v30 = v28;
      v7 += 80;
      v28 += 80;
      sub_2DD3500(v30, v29, a3, a4, a5, a6);
      *(_DWORD *)(v28 - 16) = *(_DWORD *)(v7 - 16);
      a3 = *(unsigned int *)(v7 - 8);
      *(_DWORD *)(v28 - 8) = a3;
      --v27;
    }
    while ( v27 );
    v31 = 80;
    if ( v26 > 0 )
      v31 = v26;
    v35 += v31;
  }
LABEL_13:
  v18 = a4 - v6;
  v19 = 0xCCCCCCCCCCCCCCCDLL * ((a4 - v6) >> 4);
  if ( a4 - v6 <= 0 )
    return v35;
  v20 = v35;
  do
  {
    v21 = (char **)v6;
    v22 = v20;
    v6 += 80;
    v20 += 80;
    sub_2DD3500(v22, v21, a3, a4, a5, a6);
    *(_DWORD *)(v20 - 16) = *(_DWORD *)(v6 - 16);
    *(_DWORD *)(v20 - 8) = *(_DWORD *)(v6 - 8);
    --v19;
  }
  while ( v19 );
  v23 = 80;
  if ( v18 > 0 )
    v23 = v18;
  return v35 + v23;
}
