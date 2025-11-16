// Function: sub_1BD5220
// Address: 0x1bd5220
//
__int64 __fastcall sub_1BD5220(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned int v4; // esi
  __int64 v5; // r8
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  int v12; // r15d
  __int64 *v13; // r10
  int v14; // eax
  int v15; // edx
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rsi
  int v21; // r9d
  __int64 *v22; // r8
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  int v26; // r8d
  unsigned int v27; // r13d
  __int64 *v28; // rdi
  __int64 v29; // rcx

  v3 = *(_QWORD *)a1;
  v4 = *(_DWORD *)(*(_QWORD *)a1 + 64LL);
  v5 = *(_QWORD *)a1 + 40LL;
  if ( !v4 )
  {
    ++*(_QWORD *)(v3 + 40);
    goto LABEL_17;
  }
  v6 = *(_QWORD *)(v3 + 48);
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v12 = 1;
    v13 = 0;
    while ( v9 != -8 )
    {
      if ( v9 == -16 && !v13 )
        v13 = v8;
      v7 = (v4 - 1) & (v12 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      ++v12;
    }
    if ( !v13 )
      v13 = v8;
    v14 = *(_DWORD *)(v3 + 56);
    ++*(_QWORD *)(v3 + 40);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(v3 + 60) - v15 > v4 >> 3 )
      {
LABEL_13:
        *(_DWORD *)(v3 + 56) = v15;
        if ( *v13 != -8 )
          --*(_DWORD *)(v3 + 60);
        *v13 = a2;
        v13[1] = 0;
        return 0;
      }
      sub_1BC8C30(v5, v4);
      v23 = *(_DWORD *)(v3 + 64);
      if ( v23 )
      {
        v24 = v23 - 1;
        v25 = *(_QWORD *)(v3 + 48);
        v26 = 1;
        v27 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v28 = 0;
        v15 = *(_DWORD *)(v3 + 56) + 1;
        v13 = (__int64 *)(v25 + 16LL * v27);
        v29 = *v13;
        if ( a2 != *v13 )
        {
          while ( v29 != -8 )
          {
            if ( v29 == -16 && !v28 )
              v28 = v13;
            v27 = v24 & (v26 + v27);
            v13 = (__int64 *)(v25 + 16LL * v27);
            v29 = *v13;
            if ( a2 == *v13 )
              goto LABEL_13;
            ++v26;
          }
          if ( v28 )
            v13 = v28;
        }
        goto LABEL_13;
      }
LABEL_45:
      ++*(_DWORD *)(v3 + 56);
      BUG();
    }
LABEL_17:
    sub_1BC8C30(v5, 2 * v4);
    v16 = *(_DWORD *)(v3 + 64);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(v3 + 48);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(v3 + 56) + 1;
      v13 = (__int64 *)(v18 + 16LL * v19);
      v20 = *v13;
      if ( a2 != *v13 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( !v22 && v20 == -16 )
            v22 = v13;
          v19 = v17 & (v21 + v19);
          v13 = (__int64 *)(v18 + 16LL * v19);
          v20 = *v13;
          if ( a2 == *v13 )
            goto LABEL_13;
          ++v21;
        }
        if ( v22 )
          v13 = v22;
      }
      goto LABEL_13;
    }
    goto LABEL_45;
  }
LABEL_3:
  v10 = v8[1];
  if ( !v10 || *(_DWORD *)(v10 + 80) != *(_DWORD *)(v3 + 224) )
    return 0;
  return sub_1BD4C40(a1, a2);
}
