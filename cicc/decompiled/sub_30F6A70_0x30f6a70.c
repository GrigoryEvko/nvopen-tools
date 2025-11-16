// Function: sub_30F6A70
// Address: 0x30f6a70
//
void __fastcall sub_30F6A70(__int64 a1)
{
  char **v1; // r9
  char **v2; // r14
  char *v3; // r13
  unsigned __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rdx
  __int64 v11; // rcx
  const __m128i *v12; // rdx
  __m128i *v13; // rax
  __int64 v14; // rax
  __m128i *v15; // r13
  __int64 v16; // rax
  __int64 *v17; // r14
  signed __int64 v18; // r15
  __m128i *v19; // rax
  __int64 *v20; // r12
  __int64 *v21; // rcx
  __int64 v22; // r9
  __int64 *v23; // rax
  __m128i v24; // xmm1
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 *v27; // rbx
  unsigned __int64 *v28; // r15
  unsigned __int64 v29; // r13
  __int64 v30; // rax
  unsigned __int64 v31; // r14
  unsigned __int64 v32; // r12
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // r10
  unsigned __int64 *v36; // rbx
  unsigned __int64 v37; // r13
  __int64 v38; // rax
  unsigned __int64 v39; // r14
  unsigned __int64 v40; // r12
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  __int64 v43; // [rsp+8h] [rbp-308h]
  const void *v44; // [rsp+10h] [rbp-300h]
  __int64 v45; // [rsp+18h] [rbp-2F8h]
  unsigned __int64 *v46; // [rsp+20h] [rbp-2F0h]
  __int64 v47; // [rsp+28h] [rbp-2E8h]
  _QWORD v48[4]; // [rsp+30h] [rbp-2E0h] BYREF
  unsigned __int64 *v49; // [rsp+50h] [rbp-2C0h] BYREF
  __int64 v50; // [rsp+58h] [rbp-2B8h]
  _BYTE v51[688]; // [rsp+60h] [rbp-2B0h] BYREF

  v46 = (unsigned __int64 *)v51;
  v49 = (unsigned __int64 *)v51;
  v50 = 0x800000000LL;
  if ( sub_30F5D70(a1, (__int64)&v49) )
  {
    v1 = *(char ***)a1;
    v47 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( (char **)v47 == v1 )
    {
      v14 = *(unsigned int *)(a1 + 152);
    }
    else
    {
      v2 = v1;
      v45 = a1 + 144;
      v44 = (const void *)(a1 + 160);
      do
      {
        v3 = *v2;
        v4 = sub_30F54A0(a1, *v2, (__int64)&v49);
        v7 = *(unsigned int *)(a1 + 156);
        v48[0] = v3;
        v48[1] = v4;
        v8 = *(unsigned int *)(a1 + 152);
        v48[2] = v9;
        v10 = v8 + 1;
        if ( v8 + 1 > v7 )
        {
          v35 = *(_QWORD *)(a1 + 144);
          if ( v35 > (unsigned __int64)v48 || (unsigned __int64)v48 >= v35 + 24 * v8 )
          {
            sub_C8D5F0(v45, v44, v10, 0x18u, v5, v6);
            v11 = *(_QWORD *)(a1 + 144);
            v8 = *(unsigned int *)(a1 + 152);
            v12 = (const __m128i *)v48;
          }
          else
          {
            v43 = *(_QWORD *)(a1 + 144);
            sub_C8D5F0(v45, v44, v10, 0x18u, v5, v6);
            v11 = *(_QWORD *)(a1 + 144);
            v8 = *(unsigned int *)(a1 + 152);
            v12 = (const __m128i *)((char *)v48 + v11 - v43);
          }
        }
        else
        {
          v11 = *(_QWORD *)(a1 + 144);
          v12 = (const __m128i *)v48;
        }
        ++v2;
        v13 = (__m128i *)(v11 + 24 * v8);
        *v13 = _mm_loadu_si128(v12);
        v13[1].m128i_i64[0] = v12[1].m128i_i64[0];
        v14 = (unsigned int)(*(_DWORD *)(a1 + 152) + 1);
        *(_DWORD *)(a1 + 152) = v14;
      }
      while ( (char **)v47 != v2 );
    }
    v15 = *(__m128i **)(a1 + 144);
    v16 = 24 * v14;
    v17 = &v15->m128i_i64[(unsigned __int64)v16 / 8];
    v18 = 0xAAAAAAAAAAAAAAABLL * (v16 >> 3);
    if ( v16 )
    {
      while ( 1 )
      {
        v47 = 24 * v18;
        v19 = (__m128i *)sub_2207800(24 * v18);
        v20 = (__int64 *)v19;
        if ( v19 )
          break;
        v18 >>= 1;
        if ( !v18 )
          goto LABEL_35;
      }
      v21 = &v19->m128i_i64[3 * v18];
      v22 = v47;
      *v19 = _mm_loadu_si128(v15);
      v19[1].m128i_i64[0] = v15[1].m128i_i64[0];
      v23 = &v19[1].m128i_i64[1];
      if ( v21 == v20 + 3 )
      {
        v26 = (__int64)v20;
      }
      else
      {
        do
        {
          v24 = _mm_loadu_si128((const __m128i *)(v23 - 3));
          v25 = *(v23 - 1);
          v23 += 3;
          *(__m128i *)(v23 - 3) = v24;
          *(v23 - 1) = v25;
        }
        while ( v21 != v23 );
        v26 = (__int64)v20 + v22 - 24;
      }
      v15->m128i_i64[0] = *(_QWORD *)v26;
      v15->m128i_i64[1] = *(_QWORD *)(v26 + 8);
      v15[1].m128i_i32[0] = *(_DWORD *)(v26 + 16);
      sub_30F6980(v15->m128i_i64, v17, v20, v18);
    }
    else
    {
LABEL_35:
      v20 = 0;
      sub_30F3F30((__int64)v15, v17);
    }
    j_j___libc_free_0((unsigned __int64)v20);
    v27 = v49;
    v28 = &v49[10 * (unsigned int)v50];
    if ( v49 != v28 )
    {
      do
      {
        v29 = *(v28 - 10);
        v30 = *((unsigned int *)v28 - 18);
        v28 -= 10;
        v31 = v29 + 8 * v30;
        if ( v29 != v31 )
        {
          do
          {
            v32 = *(_QWORD *)(v31 - 8);
            v31 -= 8LL;
            if ( v32 )
            {
              v33 = *(_QWORD *)(v32 + 64);
              if ( v33 != v32 + 80 )
                _libc_free(v33);
              v34 = *(_QWORD *)(v32 + 24);
              if ( v34 != v32 + 40 )
                _libc_free(v34);
              j_j___libc_free_0(v32);
            }
          }
          while ( v29 != v31 );
          v29 = *v28;
        }
        if ( (unsigned __int64 *)v29 != v28 + 2 )
          _libc_free(v29);
      }
      while ( v27 != v28 );
LABEL_26:
      v28 = v49;
    }
  }
  else
  {
    v36 = v49;
    v28 = &v49[10 * (unsigned int)v50];
    if ( v49 != v28 )
    {
      do
      {
        v37 = *(v28 - 10);
        v38 = *((unsigned int *)v28 - 18);
        v28 -= 10;
        v39 = v37 + 8 * v38;
        if ( v37 != v39 )
        {
          do
          {
            v40 = *(_QWORD *)(v39 - 8);
            v39 -= 8LL;
            if ( v40 )
            {
              v41 = *(_QWORD *)(v40 + 64);
              if ( v41 != v40 + 80 )
                _libc_free(v41);
              v42 = *(_QWORD *)(v40 + 24);
              if ( v42 != v40 + 40 )
                _libc_free(v42);
              j_j___libc_free_0(v40);
            }
          }
          while ( v37 != v39 );
          v37 = *v28;
        }
        if ( (unsigned __int64 *)v37 != v28 + 2 )
          _libc_free(v37);
      }
      while ( v36 != v28 );
      goto LABEL_26;
    }
  }
  if ( v28 != v46 )
    _libc_free((unsigned __int64)v28);
}
