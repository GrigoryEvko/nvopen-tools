// Function: sub_25D53E0
// Address: 0x25d53e0
//
__int64 __fastcall sub_25D53E0(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdx
  __int64 v6; // r9
  __int64 v7; // r8
  _QWORD *v8; // r10
  int v9; // r11d
  __int64 result; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // r13
  __int64 v14; // r15
  __int64 i; // r12
  int v16; // eax
  int v17; // edx
  __int64 v18; // r12
  int v19; // edx
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rsi
  int v23; // r11d
  _QWORD *v24; // rcx
  int v25; // edi
  int v26; // edi
  int v27; // r11d
  unsigned int v28; // eax
  __int64 v29; // rsi

  v4 = *(_DWORD *)(a2 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a2;
    goto LABEL_23;
  }
  v5 = *(_QWORD *)(a1 + 16);
  v6 = v4 - 1;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = 0;
  v9 = 1;
  result = (unsigned int)v6 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v5) >> 31) ^ (484763065 * (_DWORD)v5));
  v11 = (_QWORD *)(v7 + 8 * result);
  v12 = *v11;
  if ( *v11 == v5 )
    goto LABEL_3;
  while ( v12 != -1 )
  {
    if ( v8 || v12 != -2 )
      v11 = v8;
    result = (unsigned int)v6 & (v9 + (_DWORD)result);
    v12 = *(_QWORD *)(v7 + 8LL * (unsigned int)result);
    if ( v5 == v12 )
      goto LABEL_3;
    ++v9;
    v8 = v11;
    v11 = (_QWORD *)(v7 + 8LL * (unsigned int)result);
  }
  v16 = *(_DWORD *)(a2 + 16);
  if ( !v8 )
    v8 = v11;
  ++*(_QWORD *)a2;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v4 )
  {
LABEL_23:
    sub_A32210(a2, 2 * v4);
    v19 = *(_DWORD *)(a2 + 24);
    if ( v19 )
    {
      v20 = *(_QWORD *)(a1 + 16);
      v6 = (unsigned int)(v19 - 1);
      v7 = *(_QWORD *)(a2 + 8);
      v21 = (unsigned int)v6 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v20) >> 31) ^ (484763065 * (_DWORD)v20));
      v8 = (_QWORD *)(v7 + 8 * v21);
      v22 = *v8;
      v17 = *(_DWORD *)(a2 + 16) + 1;
      if ( *v8 == v20 )
        goto LABEL_17;
      v23 = 1;
      v24 = 0;
      while ( v22 != -1 )
      {
        if ( !v24 && v22 == -2 )
          v24 = v8;
        LODWORD(v21) = v6 & (v23 + v21);
        v8 = (_QWORD *)(v7 + 8LL * (unsigned int)v21);
        v22 = *v8;
        if ( v20 == *v8 )
          goto LABEL_17;
        ++v23;
      }
LABEL_27:
      if ( v24 )
        v8 = v24;
      goto LABEL_17;
    }
LABEL_44:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a2 + 20) - v17 <= v4 >> 3 )
  {
    sub_A32210(a2, v4);
    v25 = *(_DWORD *)(a2 + 24);
    if ( v25 )
    {
      v7 = *(_QWORD *)(a1 + 16);
      v26 = v25 - 1;
      v6 = *(_QWORD *)(a2 + 8);
      v24 = 0;
      v27 = 1;
      v28 = v26 & (((0xBF58476D1CE4E5B9LL * v7) >> 31) ^ (484763065 * v7));
      v8 = (_QWORD *)(v6 + 8LL * v28);
      v29 = *v8;
      v17 = *(_DWORD *)(a2 + 16) + 1;
      if ( v7 == *v8 )
        goto LABEL_17;
      while ( v29 != -1 )
      {
        if ( v29 == -2 && !v24 )
          v24 = v8;
        v28 = v26 & (v27 + v28);
        v8 = (_QWORD *)(v6 + 8LL * v28);
        v29 = *v8;
        if ( v7 == *v8 )
          goto LABEL_17;
        ++v27;
      }
      goto LABEL_27;
    }
    goto LABEL_44;
  }
LABEL_17:
  *(_DWORD *)(a2 + 16) = v17;
  if ( *v8 != -1 )
    --*(_DWORD *)(a2 + 20);
  v18 = *(_QWORD *)(a1 + 16);
  *v8 = v18;
  result = *(unsigned int *)(a2 + 40);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 44) )
  {
    sub_C8D5F0(a2 + 32, (const void *)(a2 + 48), result + 1, 8u, v7, v6);
    result = *(unsigned int *)(a2 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8 * result) = v18;
  ++*(_DWORD *)(a2 + 40);
LABEL_3:
  v13 = *(_QWORD *)(a1 + 192);
  v14 = a1 + 176;
  if ( v13 != v14 )
  {
    do
    {
      for ( i = *(_QWORD *)(v13 + 64); v13 + 48 != i; i = sub_220EF30(i) )
        sub_25D53E0(i + 40, a2);
      result = sub_220EF30(v13);
      v13 = result;
    }
    while ( v14 != result );
  }
  return result;
}
