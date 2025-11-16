// Function: sub_A40A70
// Address: 0xa40a70
//
__int64 __fastcall sub_A40A70(__int64 a1)
{
  _QWORD *v2; // r8
  _QWORD *i; // rcx
  int v4; // eax
  __int64 v5; // rdi
  int v6; // r9d
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r10
  __int64 *v10; // r8
  __int64 *j; // rcx
  int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rdi
  int v15; // r9d
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r10
  __int64 *v19; // rcx
  __int64 *k; // r8
  int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rdi
  int v24; // r9d
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r10
  __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rcx
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rax
  __int64 result; // rax
  int v36; // eax
  int v37; // r11d
  int v38; // eax
  int v39; // r11d
  int v40; // eax
  int v41; // r11d
  __int64 v42; // rax

  v2 = *(_QWORD **)(a1 + 120);
  for ( i = (_QWORD *)(*(_QWORD *)(a1 + 112) + 16LL * *(unsigned int *)(a1 + 536)); i != v2; i += 2 )
  {
    v4 = *(_DWORD *)(a1 + 104);
    v5 = *(_QWORD *)(a1 + 88);
    if ( v4 )
    {
      v6 = v4 - 1;
      v7 = (v4 - 1) & (((unsigned int)*i >> 9) ^ ((unsigned int)*i >> 4));
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( *i == *v8 )
      {
LABEL_4:
        *v8 = -8192;
        --*(_DWORD *)(a1 + 96);
        ++*(_DWORD *)(a1 + 100);
      }
      else
      {
        v38 = 1;
        while ( v9 != -4096 )
        {
          v39 = v38 + 1;
          v7 = v6 & (v38 + v7);
          v8 = (__int64 *)(v5 + 16LL * v7);
          v9 = *v8;
          if ( *i == *v8 )
            goto LABEL_4;
          v38 = v39;
        }
      }
    }
  }
  v10 = *(__int64 **)(a1 + 216);
  for ( j = (__int64 *)(*(_QWORD *)(a1 + 208) + 8LL * *(unsigned int *)(a1 + 540)); v10 != j; ++j )
  {
    v12 = *(_DWORD *)(a1 + 280);
    v13 = *j;
    v14 = *(_QWORD *)(a1 + 264);
    if ( v12 )
    {
      v15 = v12 - 1;
      v16 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v17 = (__int64 *)(v14 + 16LL * v16);
      v18 = *v17;
      if ( v13 == *v17 )
      {
LABEL_9:
        *v17 = -8192;
        --*(_DWORD *)(a1 + 272);
        ++*(_DWORD *)(a1 + 276);
      }
      else
      {
        v40 = 1;
        while ( v18 != -4096 )
        {
          v41 = v40 + 1;
          v16 = v15 & (v40 + v16);
          v17 = (__int64 *)(v14 + 16LL * v16);
          v18 = *v17;
          if ( v13 == *v17 )
            goto LABEL_9;
          v40 = v41;
        }
      }
    }
  }
  v19 = *(__int64 **)(a1 + 512);
  for ( k = *(__int64 **)(a1 + 520); v19 != k; ++v19 )
  {
    v21 = *(_DWORD *)(a1 + 104);
    v22 = *v19;
    v23 = *(_QWORD *)(a1 + 88);
    if ( v21 )
    {
      v24 = v21 - 1;
      v25 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v26 = (__int64 *)(v23 + 16LL * v25);
      v27 = *v26;
      if ( v22 == *v26 )
      {
LABEL_14:
        *v26 = -8192;
        --*(_DWORD *)(a1 + 96);
        ++*(_DWORD *)(a1 + 100);
      }
      else
      {
        v36 = 1;
        while ( v27 != -4096 )
        {
          v37 = v36 + 1;
          v25 = v24 & (v36 + v25);
          v26 = (__int64 *)(v23 + 16LL * v25);
          v27 = *v26;
          if ( v22 == *v26 )
            goto LABEL_14;
          v36 = v37;
        }
      }
    }
  }
  v28 = *(_QWORD *)(a1 + 112);
  v29 = *(unsigned int *)(a1 + 536);
  v30 = (*(_QWORD *)(a1 + 120) - v28) >> 4;
  if ( v29 > v30 )
  {
    sub_A40710((const __m128i **)(a1 + 112), v29 - v30);
  }
  else if ( v29 < v30 )
  {
    v31 = v28 + 16 * v29;
    if ( *(_QWORD *)(a1 + 120) != v31 )
      *(_QWORD *)(a1 + 120) = v31;
  }
  v32 = *(_QWORD *)(a1 + 208);
  v33 = *(unsigned int *)(a1 + 540);
  v34 = (*(_QWORD *)(a1 + 216) - v32) >> 3;
  if ( v33 > v34 )
  {
    sub_A408C0(a1 + 208, v33 - v34);
  }
  else if ( v33 < v34 )
  {
    v42 = v32 + 8 * v33;
    if ( *(_QWORD *)(a1 + 216) != v42 )
      *(_QWORD *)(a1 + 216) = v42;
  }
  result = *(_QWORD *)(a1 + 512);
  if ( result != *(_QWORD *)(a1 + 520) )
    *(_QWORD *)(a1 + 520) = result;
  *(_DWORD *)(a1 + 544) = 0;
  return result;
}
