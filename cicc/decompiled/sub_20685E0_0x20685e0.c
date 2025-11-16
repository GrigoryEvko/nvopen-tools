// Function: sub_20685E0
// Address: 0x20685e0
//
__int64 *__fastcall sub_20685E0(__int64 a1, __int64 *a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r13
  unsigned int v7; // r8d
  __int64 v8; // r10
  __int64 v9; // r9
  unsigned int v10; // eax
  __int64 **v11; // rdx
  __int64 *v12; // rdi
  __int64 *result; // rax
  int v14; // ebx
  __int64 *v15; // rcx
  int v16; // eax
  int v17; // edx
  __int64 *v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 *v21; // rax
  __int64 v22; // rsi
  int v23; // eax
  int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // eax
  __int64 v27; // rdi
  int v28; // r11d
  __int64 *v29; // r10
  int v30; // eax
  int v31; // esi
  __int64 v32; // r8
  __int64 *v33; // r10
  int v34; // r11d
  unsigned int v35; // eax
  __int64 *v36; // [rsp+18h] [rbp-28h] BYREF

  v5 = a1 + 8;
  v7 = *(_DWORD *)(a1 + 32);
  v36 = a2;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_18;
  }
  v8 = *(_QWORD *)(a1 + 16);
  v9 = (__int64)a2;
  v10 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (__int64 **)(v8 + 24LL * v10);
  v12 = *v11;
  if ( a2 != *v11 )
  {
    v14 = 1;
    v15 = 0;
    while ( v12 != (__int64 *)-8LL )
    {
      if ( !v15 && v12 == (__int64 *)-16LL )
        v15 = (__int64 *)v11;
      v10 = (v7 - 1) & (v14 + v10);
      v11 = (__int64 **)(v8 + 24LL * v10);
      v12 = *v11;
      if ( a2 == *v11 )
        goto LABEL_3;
      ++v14;
    }
    v16 = *(_DWORD *)(a1 + 24);
    if ( !v15 )
      v15 = (__int64 *)v11;
    ++*(_QWORD *)(a1 + 8);
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 28) - v17 > v7 >> 3 )
      {
LABEL_12:
        *(_DWORD *)(a1 + 24) = v17;
        if ( *v15 != -8 )
          --*(_DWORD *)(a1 + 28);
        v15[1] = 0;
        *((_DWORD *)v15 + 4) = 0;
        *v15 = v9;
        a2 = v36;
        goto LABEL_15;
      }
      sub_205F3F0(v5, v7);
      v30 = *(_DWORD *)(a1 + 32);
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a1 + 16);
        v33 = 0;
        v34 = 1;
        v35 = (v30 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v15 = (__int64 *)(v32 + 24LL * v35);
        v17 = *(_DWORD *)(a1 + 24) + 1;
        v9 = *v15;
        if ( v36 != (__int64 *)*v15 )
        {
          while ( v9 != -8 )
          {
            if ( !v33 && v9 == -16 )
              v33 = v15;
            v35 = v31 & (v34 + v35);
            v15 = (__int64 *)(v32 + 24LL * v35);
            v9 = *v15;
            if ( v36 == (__int64 *)*v15 )
              goto LABEL_12;
            ++v34;
          }
          v9 = (__int64)v36;
          if ( v33 )
            v15 = v33;
        }
        goto LABEL_12;
      }
LABEL_46:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_18:
    sub_205F3F0(v5, 2 * v7);
    v23 = *(_DWORD *)(a1 + 32);
    if ( v23 )
    {
      v9 = (__int64)v36;
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 16);
      v26 = (v23 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v15 = (__int64 *)(v25 + 24LL * v26);
      v17 = *(_DWORD *)(a1 + 24) + 1;
      v27 = *v15;
      if ( (__int64 *)*v15 != v36 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != -8 )
        {
          if ( !v29 && v27 == -16 )
            v29 = v15;
          v26 = v24 & (v28 + v26);
          v15 = (__int64 *)(v25 + 24LL * v26);
          v27 = *v15;
          if ( v36 == (__int64 *)*v15 )
            goto LABEL_12;
          ++v28;
        }
        if ( v29 )
          v15 = v29;
      }
      goto LABEL_12;
    }
    goto LABEL_46;
  }
LABEL_3:
  if ( v11[1] )
    return v11[1];
LABEL_15:
  result = sub_20542C0(a1, (__int64)a2, *a2, a3, a4, a5);
  if ( !result )
  {
    v18 = sub_2067630(a1, (__int64)v36, a3, a4, a5);
    v20 = v19;
    v21 = sub_205F5C0(v5, (__int64 *)&v36);
    v22 = (__int64)v36;
    v21[1] = (__int64)v18;
    *((_DWORD *)v21 + 4) = v20;
    sub_20540C0(a1, v22, (__int64)v18, v20);
    return v18;
  }
  return result;
}
