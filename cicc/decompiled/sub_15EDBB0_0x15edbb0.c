// Function: sub_15EDBB0
// Address: 0x15edbb0
//
__int64 __fastcall sub_15EDBB0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rcx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __m128i *v8; // r14
  const __m128i *i; // r12
  const __m128i *v10; // rax
  __int8 v11; // al
  __int8 v12; // al
  __int64 v13; // rax
  unsigned __int64 v14; // rbx
  __int64 v15; // rcx
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // r14
  __int64 v18; // rdx
  unsigned __int64 v19; // r12
  _QWORD *v20; // r13
  unsigned __int64 v21; // r8
  _QWORD *v22; // r12
  const __m128i *v24; // [rsp+8h] [rbp-68h]
  int v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  unsigned __int64 v27; // [rsp+28h] [rbp-48h]
  unsigned __int64 v28; // [rsp+30h] [rbp-40h]
  unsigned __int64 v29; // [rsp+38h] [rbp-38h]

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v2 = a2;
  v3 = (((((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
         | (*(unsigned int *)(a1 + 12) + 2LL)
         | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | (((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  v5 = 0xFFFFFFFFLL;
  if ( v4 >= a2 )
    v2 = v4;
  if ( v2 <= 0xFFFFFFFF )
    v5 = v2;
  v25 = v5;
  v26 = malloc(832 * v5);
  if ( !v26 )
    sub_16BD1C0("Allocation failed");
  v6 = *(unsigned int *)(a1 + 8);
  v7 = 3 * v6;
  v28 = *(_QWORD *)a1 + 832 * v6;
  if ( *(_QWORD *)a1 != v28 )
  {
    v8 = (__m128i *)v26;
    for ( i = (const __m128i *)(*(_QWORD *)a1 + 16LL); ; i += 52 )
    {
      if ( v8 )
      {
        v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
        v10 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v10 == i )
        {
          v8[1] = _mm_loadu_si128(i);
        }
        else
        {
          v8->m128i_i64[0] = (__int64)v10;
          v8[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v8->m128i_i64[1] = i[-1].m128i_i64[1];
        v11 = i[1].m128i_i8[0];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v8[2].m128i_i8[0] = v11;
        v12 = i[1].m128i_i8[1];
        v8[3].m128i_i32[0] = 0;
        v8[2].m128i_i8[1] = v12;
        v8[2].m128i_i64[1] = (__int64)&v8[3].m128i_i64[1];
        v8[3].m128i_i32[1] = 4;
        if ( i[2].m128i_i32[0] )
          sub_15ED230((__int64)&v8[2].m128i_i64[1], (__int64)&i[1].m128i_i64[1], v7);
        v8[51].m128i_i64[1] = i[50].m128i_i64[1];
      }
      v8 += 52;
      if ( (const __m128i *)v28 == &i[51] )
        break;
    }
    v24 = *(const __m128i **)a1;
    v28 = *(_QWORD *)a1 + 832LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v28 )
    {
      do
      {
        v28 -= 832LL;
        v29 = *(_QWORD *)(v28 + 40);
        v13 = 192LL * *(unsigned int *)(v28 + 48);
        v14 = v29 + v13;
        if ( v29 != v29 + v13 )
        {
          do
          {
            v15 = *(unsigned int *)(v14 - 120);
            v16 = *(_QWORD *)(v14 - 128);
            v14 -= 192LL;
            v17 = v16 + 56 * v15;
            if ( v16 != v17 )
            {
              do
              {
                v18 = *(unsigned int *)(v17 - 40);
                v19 = *(_QWORD *)(v17 - 48);
                v17 -= 56LL;
                v18 *= 32;
                v20 = (_QWORD *)(v19 + v18);
                if ( v19 != v19 + v18 )
                {
                  do
                  {
                    v20 -= 4;
                    if ( (_QWORD *)*v20 != v20 + 2 )
                      j_j___libc_free_0(*v20, v20[2] + 1LL);
                  }
                  while ( (_QWORD *)v19 != v20 );
                  v19 = *(_QWORD *)(v17 + 8);
                }
                if ( v19 != v17 + 24 )
                  _libc_free(v19);
              }
              while ( v16 != v17 );
              v16 = *(_QWORD *)(v14 + 64);
            }
            if ( v16 != v14 + 80 )
              _libc_free(v16);
            v21 = *(_QWORD *)(v14 + 16);
            v22 = (_QWORD *)(v21 + 32LL * *(unsigned int *)(v14 + 24));
            if ( (_QWORD *)v21 != v22 )
            {
              do
              {
                v22 -= 4;
                if ( (_QWORD *)*v22 != v22 + 2 )
                {
                  v27 = v21;
                  j_j___libc_free_0(*v22, v22[2] + 1LL);
                  v21 = v27;
                }
              }
              while ( (_QWORD *)v21 != v22 );
              v21 = *(_QWORD *)(v14 + 16);
            }
            if ( v21 != v14 + 32 )
              _libc_free(v21);
          }
          while ( v29 != v14 );
          v29 = *(_QWORD *)(v28 + 40);
        }
        if ( v29 != v28 + 56 )
          _libc_free(v29);
        if ( *(_QWORD *)v28 != v28 + 16 )
          j_j___libc_free_0(*(_QWORD *)v28, *(_QWORD *)(v28 + 16) + 1LL);
      }
      while ( (const __m128i *)v28 != v24 );
      v28 = *(_QWORD *)a1;
    }
  }
  if ( v28 != a1 + 16 )
    _libc_free(v28);
  *(_QWORD *)a1 = v26;
  *(_DWORD *)(a1 + 12) = v25;
  return a1;
}
