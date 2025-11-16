// Function: sub_AEFB90
// Address: 0xaefb90
//
void __fastcall sub_AEFB90(__int64 a1, __m128i *a2)
{
  __int64 v4; // rsi
  char *v5; // rcx
  __int64 v6; // rbx
  __int64 v7; // r13
  char *v8; // rax
  char *v9; // r8
  __int64 v10; // rdi
  __int64 v11; // rdx
  char *v12; // rdx
  unsigned int v13; // r14d
  int v14; // esi
  __int64 *v15; // rdx
  int v16; // eax
  __int64 v17; // rbx
  unsigned int v18; // r14d
  int v19; // eax
  __int64 v20; // r8
  int v21; // r9d
  unsigned int i; // ecx
  __int64 *v23; // r15
  bool v24; // al
  __int64 *v25; // rcx
  unsigned int v26; // eax
  __int64 v27; // rax
  __m128i v28; // xmm0
  __int64 v29; // [rsp+8h] [rbp-68h]
  int v30; // [rsp+10h] [rbp-60h]
  unsigned int v31; // [rsp+14h] [rbp-5Ch]
  __int64 *v32; // [rsp+18h] [rbp-58h]
  __m128i v33; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v34[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( !*(_DWORD *)(a1 + 16) )
  {
    v4 = *(unsigned int *)(a1 + 40);
    v5 = *(char **)(a1 + 32);
    v6 = a2->m128i_i64[0];
    v7 = a2->m128i_i64[1];
    v8 = v5;
    v9 = &v5[16 * v4];
    v10 = (16 * v4) >> 4;
    v11 = (16 * v4) >> 6;
    if ( v11 )
    {
      v12 = &v5[64 * v11];
      while ( *(_QWORD *)v8 != v6 || *((_QWORD *)v8 + 1) != v7 )
      {
        if ( *((_QWORD *)v8 + 2) == v6 && *((_QWORD *)v8 + 3) == v7 )
        {
          v8 += 16;
          break;
        }
        if ( *((_QWORD *)v8 + 4) == v6 && *((_QWORD *)v8 + 5) == v7 )
        {
          v8 += 32;
          break;
        }
        if ( *((_QWORD *)v8 + 6) == v6 && *((_QWORD *)v8 + 7) == v7 )
        {
          v8 += 48;
          break;
        }
        v8 += 64;
        if ( v12 == v8 )
        {
          v10 = (v9 - v8) >> 4;
          goto LABEL_31;
        }
      }
LABEL_10:
      if ( v9 != v8 )
        return;
      goto LABEL_34;
    }
LABEL_31:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
          goto LABEL_34;
        goto LABEL_42;
      }
      if ( *(_QWORD *)v8 == v6 && *((_QWORD *)v8 + 1) == v7 )
        goto LABEL_10;
      v8 += 16;
    }
    if ( *(_QWORD *)v8 == v6 && *((_QWORD *)v8 + 1) == v7 )
      goto LABEL_10;
    v8 += 16;
LABEL_42:
    if ( *(_QWORD *)v8 == v6 && *((_QWORD *)v8 + 1) == v7 )
      goto LABEL_10;
LABEL_34:
    if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, a1 + 48, v4 + 1, 16);
      v5 = *(char **)(a1 + 32);
      v4 = *(unsigned int *)(a1 + 40);
    }
    v25 = (__int64 *)&v5[16 * v4];
    *v25 = v6;
    v25[1] = v7;
    v26 = *(_DWORD *)(a1 + 40) + 1;
    *(_DWORD *)(a1 + 40) = v26;
    if ( v26 > 2 )
      sub_AEF900(a1);
    return;
  }
  v13 = *(_DWORD *)(a1 + 24);
  if ( v13 )
  {
    v17 = *(_QWORD *)(a1 + 8);
    v34[0] = (__int64 *)-8192LL;
    v34[1] = (__int64 *)-8192LL;
    v18 = v13 - 1;
    v19 = sub_AEA4A0(a2->m128i_i64, &a2->m128i_i64[1]);
    v20 = a2->m128i_i64[0];
    v15 = 0;
    v21 = 1;
    for ( i = v18 & v19; ; i = v18 & (v30 + v31) )
    {
      v23 = (__int64 *)(v17 + 16LL * i);
      if ( *v23 == v20 && a2->m128i_i64[1] == v23[1] )
        break;
      if ( *v23 == -4096 && v23[1] == -4096 )
      {
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
          v15 = (__int64 *)(v17 + 16LL * i);
        v16 = *(_DWORD *)(a1 + 16) + 1;
        ++*(_QWORD *)a1;
        v34[0] = v15;
        if ( 4 * v16 >= 3 * v13 )
          goto LABEL_14;
        if ( v13 - *(_DWORD *)(a1 + 20) - v16 > v13 >> 3 )
          goto LABEL_50;
        v14 = v13;
        goto LABEL_15;
      }
      v29 = v20;
      v30 = v21;
      v31 = i;
      v32 = v15;
      v33.m128i_i64[0] = (__int64)v34;
      v24 = sub_AE74A0((_QWORD *)(v17 + 16LL * i), v34);
      v20 = v29;
      if ( v32 || !v24 )
        v23 = v32;
      v15 = v23;
      v21 = v30 + 1;
    }
  }
  else
  {
    ++*(_QWORD *)a1;
    v34[0] = 0;
LABEL_14:
    v14 = 2 * v13;
LABEL_15:
    sub_AEF630(a1, v14);
    sub_AEAAD0(a1, a2->m128i_i64, v34);
    v15 = v34[0];
    v16 = *(_DWORD *)(a1 + 16) + 1;
LABEL_50:
    *(_DWORD *)(a1 + 16) = v16;
    if ( *v15 != -4096 || v15[1] != -4096 )
      --*(_DWORD *)(a1 + 20);
    *(__m128i *)v15 = _mm_loadu_si128(a2);
    v27 = *(unsigned int *)(a1 + 40);
    v28 = _mm_loadu_si128(a2);
    if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      v33 = v28;
      sub_C8D5F0(a1 + 32, a1 + 48, v27 + 1, 16);
      v27 = *(unsigned int *)(a1 + 40);
      v28 = _mm_load_si128(&v33);
    }
    *(__m128i *)(*(_QWORD *)(a1 + 32) + 16 * v27) = v28;
    ++*(_DWORD *)(a1 + 40);
  }
}
