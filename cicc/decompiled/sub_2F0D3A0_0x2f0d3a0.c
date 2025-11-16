// Function: sub_2F0D3A0
// Address: 0x2f0d3a0
//
void __fastcall sub_2F0D3A0(__int64 *a1)
{
  __m128i *v1; // r14
  size_t v2; // rdx
  __m128i v3; // xmm3
  __int64 v4; // rax
  __m128i v5; // xmm2
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __int32 v8; // edx
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  bool v11; // bl
  __m128i *v12; // rdi
  __m128i *v13; // rbx
  __m128i *v14; // rdx
  size_t v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rax
  int v20; // eax
  __int64 *v21; // [rsp+18h] [rbp-F8h]
  __int64 v22; // [rsp+20h] [rbp-F0h]
  __m128i *v23; // [rsp+28h] [rbp-E8h]
  size_t n; // [rsp+30h] [rbp-E0h]
  __m128i v25; // [rsp+38h] [rbp-D8h] BYREF
  __m128i v26; // [rsp+48h] [rbp-C8h] BYREF
  int v27; // [rsp+58h] [rbp-B8h]
  __int64 v28; // [rsp+60h] [rbp-B0h]
  _QWORD *v29; // [rsp+68h] [rbp-A8h] BYREF
  _QWORD v30[2]; // [rsp+78h] [rbp-98h] BYREF
  __m128i v31; // [rsp+88h] [rbp-88h]
  __int32 v32; // [rsp+98h] [rbp-78h]
  __int64 v33; // [rsp+A0h] [rbp-70h]
  _QWORD *v34; // [rsp+A8h] [rbp-68h] BYREF
  _QWORD v35[2]; // [rsp+B8h] [rbp-58h] BYREF
  __m128i v36; // [rsp+C8h] [rbp-48h]
  int v37; // [rsp+D8h] [rbp-38h]

  v1 = (__m128i *)(a1 + 3);
  v22 = *a1;
  v23 = &v25;
  if ( (__int64 *)a1[1] == a1 + 3 )
  {
    v25 = _mm_loadu_si128((const __m128i *)(a1 + 3));
  }
  else
  {
    v23 = (__m128i *)a1[1];
    v25.m128i_i64[0] = a1[3];
  }
  v2 = a1[2];
  a1[1] = (__int64)v1;
  v3 = _mm_loadu_si128((const __m128i *)(a1 + 5));
  a1[2] = 0;
  n = v2;
  LODWORD(v2) = *((_DWORD *)a1 + 14);
  *((_BYTE *)a1 + 24) = 0;
  v27 = v2;
  v26 = v3;
  while ( 1 )
  {
    v6 = (_BYTE *)v1[-5].m128i_i64[0];
    v21 = &v1[-2].m128i_i64[1];
    v28 = v1[-6].m128i_i64[1];
    v7 = v1[-5].m128i_i64[1];
    v29 = v30;
    sub_2F07250((__int64 *)&v29, v6, (__int64)&v6[v7]);
    v8 = v1[-2].m128i_i32[0];
    v34 = v35;
    v9 = _mm_loadu_si128(v1 - 3);
    v32 = v8;
    v31 = v9;
    v33 = v22;
    sub_2F07250((__int64 *)&v34, v23, (__int64)v23->m128i_i64 + n);
    v10 = _mm_loadu_si128(&v26);
    v37 = v27;
    v36 = v10;
    v11 = (unsigned int)v33 < (unsigned int)v28;
    if ( (_DWORD)v33 == (_DWORD)v28 )
      v11 = HIDWORD(v33) < HIDWORD(v28);
    if ( v34 != v35 )
      j_j___libc_free_0((unsigned __int64)v34);
    if ( v29 != v30 )
      j_j___libc_free_0((unsigned __int64)v29);
    v12 = (__m128i *)v1[-1].m128i_i64[0];
    if ( !v11 )
      break;
    v13 = v1 - 4;
    v1[-2].m128i_i64[1] = v1[-6].m128i_i64[1];
    v14 = (__m128i *)v1[-5].m128i_i64[0];
    if ( v14 == &v1[-4] )
    {
      v15 = v1[-5].m128i_u64[1];
      if ( v15 )
      {
        if ( v15 == 1 )
          v12->m128i_i8[0] = v13->m128i_i8[0];
        else
          memcpy(v12, &v1[-4], v15);
        v15 = v1[-5].m128i_u64[1];
        v12 = (__m128i *)v13[3].m128i_i64[0];
      }
      v13[3].m128i_i64[1] = v15;
      v12->m128i_i8[v15] = 0;
      v12 = (__m128i *)v1[-5].m128i_i64[0];
    }
    else
    {
      if ( v12 == v1 )
      {
        v16 = v1[-5].m128i_i64[1];
        v13[3].m128i_i64[0] = (__int64)v14;
        v13[3].m128i_i64[1] = v16;
        v1->m128i_i64[0] = v1[-4].m128i_i64[0];
      }
      else
      {
        v13[3].m128i_i64[0] = (__int64)v14;
        v4 = v1->m128i_i64[0];
        v1[-1].m128i_i64[1] = v1[-5].m128i_i64[1];
        v1->m128i_i64[0] = v1[-4].m128i_i64[0];
        if ( v12 )
        {
          v1[-5].m128i_i64[0] = (__int64)v12;
          v13->m128i_i64[0] = v4;
          goto LABEL_7;
        }
      }
      v1[-5].m128i_i64[0] = (__int64)v1[-4].m128i_i64;
      v12 = v1 - 4;
    }
LABEL_7:
    v1[-5].m128i_i64[1] = 0;
    v1 -= 4;
    v12->m128i_i8[0] = 0;
    v5 = _mm_loadu_si128(v13 + 1);
    v13[6].m128i_i32[0] = v13[2].m128i_i32[0];
    v13[5] = v5;
  }
  *v21 = v22;
  v17 = n;
  if ( v23 == &v25 )
  {
    if ( n )
    {
      if ( n == 1 )
      {
        v12->m128i_i8[0] = v25.m128i_i8[0];
        v17 = 1;
      }
      else
      {
        memcpy(v12, &v25, n);
        v17 = n;
      }
      v12 = (__m128i *)v21[1];
    }
    v21[2] = v17;
    v12->m128i_i8[v17] = 0;
    v12 = v23;
  }
  else
  {
    v18 = v25.m128i_i64[0];
    if ( v12 == v1 )
    {
      v21[1] = (__int64)v23;
      v21[2] = n;
      v1->m128i_i64[0] = v18;
    }
    else
    {
      v19 = v1->m128i_i64[0];
      v21[1] = (__int64)v23;
      v21[2] = n;
      v1->m128i_i64[0] = v18;
      if ( v12 )
      {
        v23 = v12;
        v25.m128i_i64[0] = v19;
        goto LABEL_27;
      }
    }
    v23 = &v25;
    v12 = &v25;
  }
LABEL_27:
  v12->m128i_i8[0] = 0;
  v20 = v27;
  *(__m128i *)(v21 + 5) = _mm_loadu_si128(&v26);
  *((_DWORD *)v21 + 14) = v20;
  if ( v23 != &v25 )
    j_j___libc_free_0((unsigned __int64)v23);
}
