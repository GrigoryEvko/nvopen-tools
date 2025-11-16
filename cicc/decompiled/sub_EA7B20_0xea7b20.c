// Function: sub_EA7B20
// Address: 0xea7b20
//
__int64 __fastcall sub_EA7B20(__int64 a1, const void *a2, size_t a3, const __m128i *a4)
{
  __int64 *v4; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __int64 v10; // rax
  __m128i v11; // xmm1
  __int8 v12; // r8
  _QWORD *v13; // r13
  _QWORD *v14; // r12
  _QWORD *v15; // rbx
  __int64 v16; // rax
  int v17; // eax
  unsigned int v18; // r10d
  _QWORD *v19; // r11
  __int64 result; // rax
  _QWORD *v21; // r14
  _QWORD *v22; // r14
  __int64 v23; // rbx
  __int64 v24; // r15
  __int64 v25; // rdi
  __int64 v26; // rax
  unsigned int v27; // r10d
  _QWORD *v28; // r11
  char *v29; // rcx
  __m128i v30; // xmm2
  __m128i v31; // xmm3
  char *src; // [rsp+0h] [rbp-E0h]
  _QWORD *v33; // [rsp+8h] [rbp-D8h]
  __int32 v34; // [rsp+10h] [rbp-D0h]
  __int8 v35; // [rsp+17h] [rbp-C9h]
  __int64 *v36; // [rsp+18h] [rbp-C8h]
  unsigned int v37; // [rsp+20h] [rbp-C0h]
  __int64 v38; // [rsp+30h] [rbp-B0h]
  __int64 v39; // [rsp+30h] [rbp-B0h]
  _QWORD *v40; // [rsp+38h] [rbp-A8h]
  __m128i v41; // [rsp+50h] [rbp-90h] BYREF
  __m128i v42[8]; // [rsp+60h] [rbp-80h] BYREF

  v4 = (__int64 *)(a1 + 2384);
  v8 = a4[4].m128i_i64[1];
  v9 = _mm_loadu_si128(a4);
  v36 = v4;
  v10 = a4[3].m128i_i64[0];
  v11 = _mm_loadu_si128(a4 + 1);
  a4[3].m128i_i64[0] = 0;
  v12 = a4[5].m128i_i8[0];
  v13 = (_QWORD *)a4[2].m128i_i64[0];
  a4[4].m128i_i64[1] = 0;
  v14 = (_QWORD *)a4[2].m128i_i64[1];
  v15 = (_QWORD *)a4[4].m128i_i64[0];
  v38 = v10;
  a4[2].m128i_i64[1] = 0;
  v16 = a4[3].m128i_i64[1];
  a4[2].m128i_i64[0] = 0;
  a4[4].m128i_i64[0] = 0;
  a4[3].m128i_i64[1] = 0;
  v35 = v12;
  v34 = a4[5].m128i_i32[1];
  v41 = v9;
  v42[0] = v11;
  v40 = (_QWORD *)v16;
  v17 = sub_C92610();
  v18 = sub_C92740(a1 + 2384, a2, a3, v17);
  v19 = (_QWORD *)(*(_QWORD *)(a1 + 2384) + 8LL * v18);
  if ( *v19 )
  {
    if ( *v19 != -8 )
    {
      result = (__int64)v40;
      v39 = v38 - (_QWORD)v13;
      if ( v40 != v15 )
      {
        v21 = v40;
        do
        {
          result = (__int64)(v21 + 2);
          if ( (_QWORD *)*v21 != v21 + 2 )
            result = j_j___libc_free_0(*v21, v21[2] + 1LL);
          v21 += 4;
        }
        while ( v21 != v15 );
      }
      if ( v40 )
        result = j_j___libc_free_0(v40, v8 - (_QWORD)v40);
      if ( v13 != v14 )
      {
        v22 = v13;
        do
        {
          v23 = v22[3];
          v24 = v22[2];
          if ( v23 != v24 )
          {
            do
            {
              if ( *(_DWORD *)(v24 + 32) > 0x40u )
              {
                v25 = *(_QWORD *)(v24 + 24);
                if ( v25 )
                  result = j_j___libc_free_0_0(v25);
              }
              v24 += 40;
            }
            while ( v23 != v24 );
            v24 = v22[2];
          }
          if ( v24 )
            result = j_j___libc_free_0(v24, v22[4] - v24);
          v22 += 6;
        }
        while ( v22 != v14 );
      }
      if ( v13 )
        return j_j___libc_free_0(v13, v39);
      return result;
    }
    --*(_DWORD *)(a1 + 2400);
  }
  v33 = v19;
  v37 = v18;
  v26 = sub_C7D670(a3 + 97, 8);
  v27 = v37;
  v28 = v33;
  v29 = (char *)v26;
  if ( a3 )
  {
    src = (char *)v26;
    memcpy((void *)(v26 + 96), a2, a3);
    v27 = v37;
    v28 = v33;
    v29 = src;
  }
  v29[a3 + 96] = 0;
  v30 = _mm_loadu_si128(&v41);
  v31 = _mm_loadu_si128(v42);
  *(_QWORD *)v29 = a3;
  *((_QWORD *)v29 + 7) = v38;
  *((_QWORD *)v29 + 5) = v13;
  *((_QWORD *)v29 + 8) = v40;
  *((_QWORD *)v29 + 6) = v14;
  *((_QWORD *)v29 + 10) = v8;
  *((_QWORD *)v29 + 9) = v15;
  v29[88] = v35;
  *(__m128i *)(v29 + 8) = v30;
  *((_DWORD *)v29 + 23) = v34;
  *(__m128i *)(v29 + 24) = v31;
  *v28 = v29;
  ++*(_DWORD *)(a1 + 2396);
  return sub_C929D0(v36, v27);
}
