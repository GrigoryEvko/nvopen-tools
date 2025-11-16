// Function: sub_1DA9020
// Address: 0x1da9020
//
__int64 __fastcall sub_1DA9020(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v6; // esi
  __int64 v7; // r9
  unsigned int v8; // ecx
  int *v9; // r8
  int v10; // eax
  __int64 v11; // rcx
  __int64 result; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 i; // rcx
  int v17; // r15d
  int *v18; // r11
  int v19; // eax
  int v20; // ecx
  int v21; // eax
  int v22; // esi
  __int64 v23; // r9
  unsigned int v24; // eax
  int v25; // edi
  int v26; // r11d
  int *v27; // r10
  int v28; // eax
  int v29; // eax
  __int64 v30; // r13
  __int64 v31; // r9
  int v32; // edi
  int v33; // esi
  int *v34; // r14
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  v5 = a1 + 232;
  v6 = *(_DWORD *)(a1 + 256);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 232);
    goto LABEL_25;
  }
  v7 = *(_QWORD *)(a1 + 240);
  v8 = (v6 - 1) & (37 * a2);
  v9 = (int *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( a2 == *v9 )
  {
    v11 = *((_QWORD *)v9 + 1);
    goto LABEL_4;
  }
  v17 = 1;
  v18 = 0;
  while ( v10 != -1 )
  {
    if ( v18 || v10 != -2 )
      v9 = v18;
    v8 = (v6 - 1) & (v17 + v8);
    v34 = (int *)(v7 + 16LL * v8);
    v10 = *v34;
    if ( *v34 == a2 )
    {
      v11 = *((_QWORD *)v34 + 1);
      v9 = v34;
      goto LABEL_4;
    }
    ++v17;
    v18 = v9;
    v9 = (int *)(v7 + 16LL * v8);
  }
  v19 = *(_DWORD *)(a1 + 248);
  if ( v18 )
    v9 = v18;
  ++*(_QWORD *)(a1 + 232);
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v6 )
  {
LABEL_25:
    v35 = a3;
    sub_1DA86E0(v5, 2 * v6);
    v21 = *(_DWORD *)(a1 + 256);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 240);
      a3 = v35;
      v24 = (v21 - 1) & (37 * a2);
      v20 = *(_DWORD *)(a1 + 248) + 1;
      v9 = (int *)(v23 + 16LL * v24);
      v25 = *v9;
      if ( *v9 == a2 )
        goto LABEL_21;
      v26 = 1;
      v27 = 0;
      while ( v25 != -1 )
      {
        if ( !v27 && v25 == -2 )
          v27 = v9;
        v24 = v22 & (v26 + v24);
        v9 = (int *)(v23 + 16LL * v24);
        v25 = *v9;
        if ( *v9 == a2 )
          goto LABEL_21;
        ++v26;
      }
LABEL_29:
      if ( v27 )
        v9 = v27;
      goto LABEL_21;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 248);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 252) - v20 <= v6 >> 3 )
  {
    v36 = a3;
    sub_1DA86E0(v5, v6);
    v28 = *(_DWORD *)(a1 + 256);
    if ( v28 )
    {
      v29 = v28 - 1;
      v27 = 0;
      a3 = v36;
      LODWORD(v30) = v29 & (37 * a2);
      v31 = *(_QWORD *)(a1 + 240);
      v20 = *(_DWORD *)(a1 + 248) + 1;
      v32 = 1;
      v9 = (int *)(v31 + 16LL * (unsigned int)v30);
      v33 = *v9;
      if ( *v9 == a2 )
        goto LABEL_21;
      while ( v33 != -1 )
      {
        if ( v33 == -2 && !v27 )
          v27 = v9;
        v30 = v29 & (unsigned int)(v30 + v32);
        v9 = (int *)(v31 + 16 * v30);
        v33 = *v9;
        if ( *v9 == a2 )
          goto LABEL_21;
        ++v32;
      }
      goto LABEL_29;
    }
    goto LABEL_51;
  }
LABEL_21:
  *(_DWORD *)(a1 + 248) = v20;
  if ( *v9 != -1 )
    --*(_DWORD *)(a1 + 252);
  *v9 = a2;
  v11 = 0;
  *((_QWORD *)v9 + 1) = 0;
LABEL_4:
  result = *(_QWORD *)(a3 + 24);
  do
  {
    v13 = result;
    result = *(_QWORD *)(result + 24);
  }
  while ( v13 != result );
  *(_QWORD *)(a3 + 24) = result;
  if ( v11 )
  {
    v14 = *(_QWORD *)(v11 + 24);
    do
    {
      v15 = v14;
      v14 = *(_QWORD *)(v14 + 24);
    }
    while ( v15 != v14 );
    *(_QWORD *)(v11 + 24) = v14;
    if ( v13 != v14 )
    {
      for ( i = *(_QWORD *)(result + 32); i; i = *(_QWORD *)(i + 32) )
      {
        *(_QWORD *)(result + 24) = v15;
        result = i;
      }
      *(_QWORD *)(result + 24) = v15;
      *(_QWORD *)(result + 32) = *(_QWORD *)(v15 + 32);
      *(_QWORD *)(v15 + 32) = v13;
    }
  }
  else
  {
    v14 = result;
  }
  *((_QWORD *)v9 + 1) = v14;
  return result;
}
