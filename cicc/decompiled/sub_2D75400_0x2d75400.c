// Function: sub_2D75400
// Address: 0x2d75400
//
void __fastcall sub_2D75400(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  char *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 *v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdx
  char *v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // r9
  unsigned int v18; // esi
  int v19; // eax
  __int64 *v20; // rdx
  int v21; // eax
  __int64 v22; // r8
  __int64 v23; // rax
  __m128i v24; // xmm0
  __m128i v25; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v26; // [rsp+10h] [rbp-30h] BYREF
  __int64 *v27; // [rsp+18h] [rbp-28h] BYREF

  if ( !*(_DWORD *)(a1 + 16) )
  {
    v8 = *(unsigned int *)(a1 + 40);
    v9 = *(char **)(a1 + 32);
    v10 = a2->m128i_i64[0];
    v11 = a2->m128i_i64[1];
    v12 = (__int64 *)&v9[16 * v8];
    v13 = (16 * v8) >> 4;
    v14 = (16 * v8) >> 6;
    if ( v14 )
    {
      v15 = &v9[64 * v14];
      while ( *(_QWORD *)v9 != v10 || *((_QWORD *)v9 + 1) != v11 )
      {
        if ( *((_QWORD *)v9 + 2) == v10 && *((_QWORD *)v9 + 3) == v11 )
        {
          if ( v12 != (__int64 *)(v9 + 16) )
            return;
          goto LABEL_15;
        }
        if ( *((_QWORD *)v9 + 4) == v10 && *((_QWORD *)v9 + 5) == v11 )
        {
          if ( v12 != (__int64 *)(v9 + 32) )
            return;
          goto LABEL_15;
        }
        if ( *((_QWORD *)v9 + 6) == v10 && *((_QWORD *)v9 + 7) == v11 )
        {
          if ( v12 != (__int64 *)(v9 + 48) )
            return;
          goto LABEL_15;
        }
        v9 += 64;
        if ( v15 == v9 )
        {
          v13 = ((char *)v12 - v9) >> 4;
          goto LABEL_10;
        }
      }
LABEL_20:
      if ( v12 != (__int64 *)v9 )
        return;
      goto LABEL_15;
    }
LABEL_10:
    if ( v13 != 2 )
    {
      if ( v13 != 3 )
      {
        if ( v13 != 1 )
          goto LABEL_15;
        goto LABEL_13;
      }
      if ( *(_QWORD *)v9 == v10 && *((_QWORD *)v9 + 1) == v11 )
        goto LABEL_20;
      v9 += 16;
    }
    if ( *(_QWORD *)v9 == v10 && *((_QWORD *)v9 + 1) == v11 )
      goto LABEL_20;
    v9 += 16;
LABEL_13:
    if ( *(_QWORD *)v9 == v10 && *((_QWORD *)v9 + 1) == v11 )
      goto LABEL_20;
LABEL_15:
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 0x10u, a5, a6);
      v12 = (__int64 *)(*(_QWORD *)(a1 + 32) + 16LL * *(unsigned int *)(a1 + 40));
    }
    *v12 = v10;
    v12[1] = v11;
    v16 = *(_DWORD *)(a1 + 40) + 1;
    *(_DWORD *)(a1 + 40) = v16;
    if ( v16 > 8 )
      sub_2D75210(a1);
    return;
  }
  if ( (unsigned __int8)sub_2D6B120(a1, a2->m128i_i64, &v26) )
    return;
  v18 = *(_DWORD *)(a1 + 24);
  v19 = *(_DWORD *)(a1 + 16);
  v20 = v26;
  ++*(_QWORD *)a1;
  v21 = v19 + 1;
  v22 = 2 * v18;
  v27 = v20;
  if ( 4 * v21 >= 3 * v18 )
  {
    v18 *= 2;
  }
  else if ( v18 - *(_DWORD *)(a1 + 20) - v21 > v18 >> 3 )
  {
    goto LABEL_34;
  }
  sub_2D74F50(a1, v18);
  sub_2D6B120(a1, a2->m128i_i64, &v27);
  v20 = v27;
  v21 = *(_DWORD *)(a1 + 16) + 1;
LABEL_34:
  *(_DWORD *)(a1 + 16) = v21;
  if ( *v20 != -4096 || v20[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v20 = a2->m128i_i64[0];
  v20[1] = a2->m128i_i64[1];
  v23 = *(unsigned int *)(a1 + 40);
  v24 = _mm_loadu_si128(a2);
  if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v25 = v24;
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v23 + 1, 0x10u, v22, v17);
    v23 = *(unsigned int *)(a1 + 40);
    v24 = _mm_load_si128(&v25);
  }
  *(__m128i *)(*(_QWORD *)(a1 + 32) + 16 * v23) = v24;
  ++*(_DWORD *)(a1 + 40);
}
