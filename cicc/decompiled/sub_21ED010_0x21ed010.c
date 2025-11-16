// Function: sub_21ED010
// Address: 0x21ed010
//
__int64 __fastcall sub_21ED010(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // rdx
  int v5; // ecx
  __int64 v6; // r8
  unsigned int v7; // edi
  __int64 *v8; // rax
  __int64 v9; // r10
  unsigned int v10; // esi
  __int64 v11; // r12
  unsigned int v12; // r9d
  __int64 v13; // r8
  unsigned int v14; // eax
  int *v15; // rdx
  int v16; // edi
  unsigned int v17; // eax
  char v18; // cl
  __int64 v19; // rax
  int v20; // eax
  int v21; // r11d
  int v22; // r10d
  unsigned int v23; // r11d
  int i; // r13d
  int v25; // r11d
  int *v26; // r10
  int v27; // eax
  int v28; // edx
  int v29; // [rsp-34h] [rbp-34h] BYREF
  int *v30; // [rsp-30h] [rbp-30h] BYREF

  result = 0;
  if ( **(_QWORD **)a1 == a2 )
    return result;
  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(unsigned int *)(v3 + 136);
  v5 = **(_DWORD **)(a1 + 8);
  v6 = *(_QWORD *)(v3 + 120);
  if ( (_DWORD)v4 )
  {
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_4:
      v10 = *(_DWORD *)(v3 + 80);
      v11 = v8[1];
      v29 = v5;
      if ( v10 )
        goto LABEL_5;
      return 0;
    }
    v20 = 1;
    while ( v9 != -8 )
    {
      v21 = v20 + 1;
      v7 = (v4 - 1) & (v20 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_4;
      v20 = v21;
    }
  }
  v10 = *(_DWORD *)(v3 + 80);
  v29 = v5;
  v11 = *(_QWORD *)(v6 + 16 * v4 + 8);
  if ( !v10 )
    return 0;
LABEL_5:
  v12 = v10 - 1;
  v13 = *(_QWORD *)(v3 + 64);
  v14 = (v10 - 1) & (37 * v5);
  v15 = (int *)(v13 + 8LL * v14);
  v16 = *v15;
  if ( v5 != *v15 )
  {
    v22 = *v15;
    v23 = (v10 - 1) & (37 * v5);
    for ( i = 1; ; ++i )
    {
      if ( v22 == -1 )
        return 0;
      v23 = v12 & (i + v23);
      v22 = *(_DWORD *)(v13 + 8LL * v23);
      if ( v5 == v22 )
        break;
    }
    v25 = 1;
    v26 = 0;
    while ( v16 != -1 )
    {
      if ( !v26 && v16 == -2 )
        v26 = v15;
      v14 = v12 & (v25 + v14);
      v15 = (int *)(v13 + 8LL * v14);
      v16 = *v15;
      if ( v5 == *v15 )
        goto LABEL_6;
      ++v25;
    }
    v27 = *(_DWORD *)(v3 + 72);
    if ( !v26 )
      v26 = v15;
    ++*(_QWORD *)(v3 + 56);
    v28 = v27 + 1;
    if ( 4 * (v27 + 1) >= 3 * v10 )
    {
      v10 *= 2;
    }
    else if ( v10 - *(_DWORD *)(v3 + 76) - v28 > v10 >> 3 )
    {
LABEL_25:
      *(_DWORD *)(v3 + 72) = v28;
      if ( *v26 != -1 )
        --*(_DWORD *)(v3 + 76);
      *v26 = v5;
      v19 = 0;
      v18 = 0;
      v26[1] = 0;
      goto LABEL_7;
    }
    sub_1BFDD60(v3 + 56, v10);
    sub_1BFD720(v3 + 56, &v29, &v30);
    v26 = v30;
    v5 = v29;
    v28 = *(_DWORD *)(v3 + 72) + 1;
    goto LABEL_25;
  }
LABEL_6:
  v17 = v15[1];
  v18 = v17 & 0x3F;
  v19 = 8LL * (v17 >> 6);
LABEL_7:
  result = (*(_QWORD *)(*(_QWORD *)(v11 + 24) + v19) >> v18) & 1LL;
  if ( !(_DWORD)result )
    return 0;
  return result;
}
