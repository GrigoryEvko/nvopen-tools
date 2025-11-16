// Function: sub_29EC8C0
// Address: 0x29ec8c0
//
void __fastcall sub_29EC8C0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  _QWORD *v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // r13
  _QWORD *v12; // rax
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 v15; // rdx
  _QWORD *v16; // rdx
  unsigned int v17; // ebx
  int v18; // esi
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 *v21; // rcx
  int v22; // eax
  __int64 v23; // r14
  unsigned int v24; // ebx
  int v25; // eax
  __int64 v26; // rdi
  unsigned int i; // edx
  __int64 *v28; // rax
  __int64 v29; // rsi
  int v30; // edx
  __int64 *v31; // rcx
  unsigned int v32; // eax
  int v33; // edx
  __int64 v34; // rax
  __m128i v35; // xmm0
  __m128i v36; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v37; // [rsp+18h] [rbp-28h] BYREF

  if ( !*(_DWORD *)(a1 + 16) )
  {
    v8 = *(unsigned int *)(a1 + 40);
    v9 = *(_QWORD **)(a1 + 32);
    v10 = a2->m128i_i64[0];
    v11 = a2->m128i_i64[1];
    v12 = v9;
    v13 = (__int64)&v9[2 * v8];
    v14 = (16 * v8) >> 4;
    v15 = (16 * v8) >> 6;
    if ( v15 )
    {
      v16 = &v9[8 * v15];
      while ( *v12 != v10 || v12[1] != v11 )
      {
        if ( v12[2] == v10 && v12[3] == v11 )
        {
          v12 += 2;
          break;
        }
        if ( v12[4] == v10 && v12[5] == v11 )
        {
          v12 += 4;
          break;
        }
        if ( v12[6] == v10 && v12[7] == v11 )
        {
          v12 += 6;
          break;
        }
        v12 += 8;
        if ( v12 == v16 )
        {
          v14 = (v13 - (__int64)v12) >> 4;
          goto LABEL_33;
        }
      }
LABEL_10:
      if ( (_QWORD *)v13 != v12 )
        return;
      goto LABEL_36;
    }
LABEL_33:
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          goto LABEL_36;
        goto LABEL_44;
      }
      if ( *v12 == v10 && v12[1] == v11 )
        goto LABEL_10;
      v12 += 2;
    }
    if ( *v12 == v10 && v12[1] == v11 )
      goto LABEL_10;
    v12 += 2;
LABEL_44:
    if ( *v12 == v10 && v12[1] == v11 )
      goto LABEL_10;
LABEL_36:
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 0x10u, v13, a6);
      v9 = *(_QWORD **)(a1 + 32);
      v8 = *(unsigned int *)(a1 + 40);
    }
    v31 = &v9[2 * v8];
    *v31 = v10;
    v31[1] = v11;
    v32 = *(_DWORD *)(a1 + 40) + 1;
    *(_DWORD *)(a1 + 40) = v32;
    if ( v32 > 2 )
      sub_AEF900(a1);
    return;
  }
  v17 = *(_DWORD *)(a1 + 24);
  if ( v17 )
  {
    v23 = *(_QWORD *)(a1 + 8);
    v24 = v17 - 1;
    v25 = sub_AEA4A0(a2->m128i_i64, &a2->m128i_i64[1]);
    v26 = a2->m128i_i64[0];
    v19 = 1;
    v21 = 0;
    for ( i = v24 & v25; ; i = v24 & v30 )
    {
      v28 = (__int64 *)(v23 + 16LL * i);
      v29 = *v28;
      if ( *v28 == v26 )
      {
        v20 = v28[1];
        if ( a2->m128i_i64[1] == v20 )
          break;
      }
      if ( v29 == -4096 )
      {
        if ( v28[1] == -4096 )
        {
          v33 = *(_DWORD *)(a1 + 16);
          v17 = *(_DWORD *)(a1 + 24);
          if ( !v21 )
            v21 = v28;
          v22 = v33 + 1;
          ++*(_QWORD *)a1;
          v37 = v21;
          if ( 4 * (v33 + 1) >= 3 * v17 )
            goto LABEL_14;
          if ( v17 - *(_DWORD *)(a1 + 20) - v22 > v17 >> 3 )
            goto LABEL_52;
          v18 = v17;
          goto LABEL_15;
        }
      }
      else if ( v29 == -8192 && v28[1] == -8192 && !v21 )
      {
        v21 = (__int64 *)(v23 + 16LL * i);
      }
      v30 = v19 + i;
      v19 = (unsigned int)(v19 + 1);
    }
  }
  else
  {
    ++*(_QWORD *)a1;
    v37 = 0;
LABEL_14:
    v18 = 2 * v17;
LABEL_15:
    sub_AEF630(a1, v18);
    sub_AEAAD0(a1, a2->m128i_i64, &v37);
    v21 = v37;
    v22 = *(_DWORD *)(a1 + 16) + 1;
LABEL_52:
    *(_DWORD *)(a1 + 16) = v22;
    if ( *v21 != -4096 || v21[1] != -4096 )
      --*(_DWORD *)(a1 + 20);
    *(__m128i *)v21 = _mm_loadu_si128(a2);
    v34 = *(unsigned int *)(a1 + 40);
    v35 = _mm_loadu_si128(a2);
    if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      v36 = v35;
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v34 + 1, 0x10u, v19, v20);
      v34 = *(unsigned int *)(a1 + 40);
      v35 = _mm_load_si128(&v36);
    }
    *(__m128i *)(*(_QWORD *)(a1 + 32) + 16 * v34) = v35;
    ++*(_DWORD *)(a1 + 40);
  }
}
