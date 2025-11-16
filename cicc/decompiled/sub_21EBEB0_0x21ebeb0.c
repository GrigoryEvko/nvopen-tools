// Function: sub_21EBEB0
// Address: 0x21ebeb0
//
__int64 __fastcall sub_21EBEB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rcx
  int v4; // edi
  __int64 v5; // r9
  unsigned int v6; // r8d
  __int64 *v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // esi
  __int64 v10; // r12
  __int64 result; // rax
  unsigned int v12; // r9d
  __int64 v13; // r8
  unsigned int v14; // eax
  int *v15; // rdx
  int v16; // ecx
  unsigned int v17; // eax
  char v18; // cl
  __int64 v19; // rax
  int v20; // eax
  int v21; // r10d
  unsigned int v22; // r11d
  int i; // r13d
  int v24; // r11d
  int v25; // r11d
  int *v26; // r10
  int v27; // eax
  int v28; // edx
  int *v29; // r13
  unsigned int v30; // eax
  int v31; // [rsp+4h] [rbp-2Ch] BYREF
  int *v32; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(unsigned int *)(v2 + 136);
  v4 = **(_DWORD **)a1;
  v5 = *(_QWORD *)(v2 + 120);
  if ( (_DWORD)v3 )
  {
    v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
      goto LABEL_3;
    v20 = 1;
    while ( v8 != -8 )
    {
      v24 = v20 + 1;
      v6 = (v3 - 1) & (v20 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v20 = v24;
    }
  }
  v7 = (__int64 *)(v5 + 16 * v3);
LABEL_3:
  v9 = *(_DWORD *)(v2 + 80);
  v10 = v7[1];
  v31 = v4;
  result = 0;
  if ( !v9 )
    return result;
  v12 = v9 - 1;
  v13 = *(_QWORD *)(v2 + 64);
  v14 = (v9 - 1) & (37 * v4);
  v15 = (int *)(v13 + 8LL * v14);
  v16 = *v15;
  if ( v4 != *v15 )
  {
    v21 = *v15;
    v22 = (v9 - 1) & (37 * v4);
    for ( i = 1; ; ++i )
    {
      if ( v21 == -1 )
        return 0;
      v22 = v12 & (i + v22);
      v21 = *(_DWORD *)(v13 + 8LL * v22);
      if ( v4 == v21 )
        break;
    }
    v25 = 1;
    v26 = 0;
    while ( v16 != -1 )
    {
      if ( v26 || v16 != -2 )
        v15 = v26;
      v14 = v12 & (v25 + v14);
      v29 = (int *)(v13 + 8LL * v14);
      v16 = *v29;
      if ( v4 == *v29 )
      {
        v30 = v29[1];
        v18 = v30 & 0x3F;
        v19 = 8LL * (v30 >> 6);
        return (*(_QWORD *)(*(_QWORD *)(v10 + 24) + v19) >> v18) & 1LL;
      }
      ++v25;
      v26 = v15;
      v15 = (int *)(v13 + 8LL * v14);
    }
    v27 = *(_DWORD *)(v2 + 72);
    if ( !v26 )
      v26 = v15;
    ++*(_QWORD *)(v2 + 56);
    v28 = v27 + 1;
    if ( 4 * (v27 + 1) >= 3 * v9 )
    {
      v9 *= 2;
    }
    else if ( v9 - *(_DWORD *)(v2 + 76) - v28 > v9 >> 3 )
    {
LABEL_23:
      *(_DWORD *)(v2 + 72) = v28;
      if ( *v26 != -1 )
        --*(_DWORD *)(v2 + 76);
      *v26 = v4;
      v18 = 0;
      v19 = 0;
      v26[1] = 0;
      return (*(_QWORD *)(*(_QWORD *)(v10 + 24) + v19) >> v18) & 1LL;
    }
    sub_1BFDD60(v2 + 56, v9);
    sub_1BFD720(v2 + 56, &v31, &v32);
    v26 = v32;
    v4 = v31;
    v28 = *(_DWORD *)(v2 + 72) + 1;
    goto LABEL_23;
  }
  v17 = v15[1];
  v18 = v17 & 0x3F;
  v19 = 8LL * (v17 >> 6);
  return (*(_QWORD *)(*(_QWORD *)(v10 + 24) + v19) >> v18) & 1LL;
}
