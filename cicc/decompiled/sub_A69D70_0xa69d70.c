// Function: sub_A69D70
// Address: 0xa69d70
//
__int64 __fastcall sub_A69D70(__int64 a1)
{
  __int64 v2; // rcx
  int v3; // esi
  _QWORD *v4; // rdi
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // rbx
  __int64 *v8; // r13
  __m128i *v9; // rcx
  __m128i *v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // r15d
  __int64 v16; // r13
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 v19; // r12
  __int64 v20; // r12
  __int64 v21; // rbx
  __m128i *v23; // r13
  __int64 v24; // rbx
  unsigned __int64 v25; // rax
  __m128i *v26; // rbx
  __m128i *v27; // rdi
  const void **v28; // rbx
  int v29; // r12d
  const void *v30; // r15
  size_t v31; // r13
  const void *v32; // rdi
  unsigned int v33; // eax
  unsigned int v34; // r10d
  __int64 *v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned int v38; // r10d
  __int64 *v39; // rcx
  __int64 v40; // r11
  __int64 *v41; // rax
  __int64 *v42; // rax
  __int64 v43; // [rsp+8h] [rbp-88h]
  __m128i *v44; // [rsp+10h] [rbp-80h]
  __int64 *v45; // [rsp+18h] [rbp-78h]
  unsigned int v46; // [rsp+24h] [rbp-6Ch]
  __int64 v47; // [rsp+28h] [rbp-68h]
  __m128i v48; // [rsp+30h] [rbp-60h] BYREF
  void *src; // [rsp+40h] [rbp-50h] BYREF
  __m128i *v50; // [rsp+48h] [rbp-48h]
  __m128i *v51; // [rsp+50h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 96);
  src = 0;
  v50 = 0;
  v3 = *(_DWORD *)(v2 + 56);
  v51 = 0;
  if ( v3 )
  {
    v4 = *(_QWORD **)(v2 + 48);
    if ( *v4 && *v4 != -8 )
    {
      v7 = *(__int64 **)(v2 + 48);
    }
    else
    {
      v5 = v4 + 1;
      do
      {
        do
        {
          v6 = *v5;
          v7 = v5++;
        }
        while ( !v6 );
      }
      while ( v6 == -8 );
    }
    v8 = &v4[v3];
    if ( v8 != v7 )
    {
      v9 = 0;
      v10 = 0;
      while ( 1 )
      {
        v11 = *(_QWORD *)*v7;
        v48.m128i_i64[0] = *v7 + 32;
        v48.m128i_i64[1] = v11;
        if ( v9 == v10 )
        {
          sub_A04210((const __m128i **)&src, v10, &v48);
          v10 = v50;
        }
        else
        {
          if ( v10 )
          {
            *v10 = _mm_loadu_si128(&v48);
            v10 = v50;
          }
          v50 = ++v10;
        }
        v12 = v7[1];
        if ( v12 && v12 != -8 )
        {
          ++v7;
        }
        else
        {
          v13 = v7 + 2;
          do
          {
            do
            {
              v14 = *v13;
              v7 = v13++;
            }
            while ( v14 == -8 );
          }
          while ( !v14 );
        }
        if ( v7 == v8 )
          break;
        v9 = v51;
      }
      v23 = (__m128i *)src;
      if ( v10 != src )
      {
        v24 = (char *)v10 - (_BYTE *)src;
        _BitScanReverse64(&v25, ((char *)v10 - (_BYTE *)src) >> 4);
        sub_A69B20((__m128i *)src, v10, 2LL * (int)(63 - (v25 ^ 0x3F)));
        if ( v24 <= 256 )
        {
          sub_A3B670(v23, v10);
        }
        else
        {
          v26 = v23 + 16;
          sub_A3B670(v23, v23 + 16);
          if ( &v23[16] != v10 )
          {
            do
            {
              v27 = v26++;
              sub_A3B600(v27);
            }
            while ( v10 != v26 );
          }
        }
        v28 = (const void **)src;
        v44 = v50;
        if ( v50 != src )
        {
          v47 = a1 + 264;
          while ( 1 )
          {
            v29 = *(_DWORD *)(a1 + 288);
            v30 = *v28;
            v31 = (size_t)v28[1];
            v32 = *v28;
            *(_DWORD *)(a1 + 288) = v29 + 1;
            v33 = sub_C92610(v32, v31);
            v34 = sub_C92740(v47, v30, v31, v33);
            v35 = (__int64 *)(*(_QWORD *)(a1 + 264) + 8LL * v34);
            v36 = *v35;
            if ( !*v35 )
              goto LABEL_44;
            if ( v36 == -8 )
              break;
LABEL_40:
            *(_DWORD *)(v36 + 8) = v29;
            v28 += 2;
            if ( v44 == (__m128i *)v28 )
              goto LABEL_20;
          }
          --*(_DWORD *)(a1 + 280);
LABEL_44:
          v45 = v35;
          v46 = v34;
          v37 = sub_C7D670(v31 + 17, 8);
          v38 = v46;
          v39 = v45;
          v40 = v37;
          if ( v31 )
          {
            v43 = v37;
            memcpy((void *)(v37 + 16), v30, v31);
            v38 = v46;
            v39 = v45;
            v40 = v43;
          }
          *(_BYTE *)(v40 + v31 + 16) = 0;
          *(_QWORD *)v40 = v31;
          *(_DWORD *)(v40 + 8) = 0;
          *v39 = v40;
          ++*(_DWORD *)(a1 + 276);
          v41 = (__int64 *)(*(_QWORD *)(a1 + 264) + 8LL * (unsigned int)sub_C929D0(v47, v38));
          v36 = *v41;
          if ( !*v41 || v36 == -8 )
          {
            v42 = v41 + 1;
            do
            {
              do
                v36 = *v42++;
              while ( v36 == -8 );
            }
            while ( !v36 );
          }
          goto LABEL_40;
        }
      }
LABEL_20:
      v2 = *(_QWORD *)(a1 + 96);
    }
  }
  v15 = *(_DWORD *)(a1 + 288);
  v16 = v2 + 8;
  v17 = v2;
  *(_DWORD *)(a1 + 328) = v15;
  v18 = *(_QWORD *)(v2 + 24);
  if ( v2 + 8 != v18 )
  {
    do
    {
      sub_A594C0(a1, *(_QWORD *)(v18 + 32));
      v18 = sub_220EF30(v18);
    }
    while ( v16 != v18 );
    v15 = *(_DWORD *)(a1 + 328);
    v17 = *(_QWORD *)(a1 + 96);
  }
  *(_DWORD *)(a1 + 392) = v15;
  v19 = *(_QWORD *)(v17 + 280);
  if ( v17 + 264 != v19 )
  {
    do
    {
      sub_A56630(a1, *(const void **)(v19 + 32), *(_QWORD *)(v19 + 40));
      v19 = sub_220EF30(v19);
    }
    while ( v17 + 264 != v19 );
    v15 = *(_DWORD *)(a1 + 392);
    v17 = *(_QWORD *)(a1 + 96);
  }
  *(_DWORD *)(a1 + 360) = v15;
  v20 = *(_QWORD *)(v17 + 232);
  v21 = v17 + 216;
  if ( v21 != v20 )
  {
    do
    {
      sub_A56500(a1, *(const void **)(v20 + 40), *(_QWORD *)(v20 + 48));
      v20 = sub_220EF30(v20);
    }
    while ( v21 != v20 );
    v15 = *(_DWORD *)(a1 + 360);
  }
  if ( src )
    j_j___libc_free_0(src, (char *)v51 - (_BYTE *)src);
  return v15;
}
