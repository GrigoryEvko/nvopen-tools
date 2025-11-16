// Function: sub_277B500
// Address: 0x277b500
//
__int64 __fastcall sub_277B500(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __m128i v3; // xmm1
  __m128i v4; // xmm2
  __m128i v5; // xmm3
  __m128i v6; // xmm4
  unsigned int v7; // eax
  _QWORD **v8; // r13
  _QWORD **i; // r12
  __int64 v10; // rax
  _QWORD *v11; // rbx
  unsigned __int64 v12; // r15
  __int64 v13; // rdi
  unsigned int v14; // eax
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  __int64 v17; // rdi
  __m128i v19; // xmm5
  __m128i v20; // xmm6
  __m128i v21; // xmm7
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __m128i v24; // [rsp+0h] [rbp-E0h] BYREF
  __m128i v25; // [rsp+10h] [rbp-D0h] BYREF
  __m128i v26; // [rsp+20h] [rbp-C0h] BYREF
  __m128i v27; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v28; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE v29[8]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD *v30; // [rsp+58h] [rbp-88h]
  unsigned int v31; // [rsp+68h] [rbp-78h]
  __int64 v32; // [rsp+78h] [rbp-68h]
  unsigned int v33; // [rsp+88h] [rbp-58h]
  __int64 v34; // [rsp+98h] [rbp-48h]
  unsigned int v35; // [rsp+A8h] [rbp-38h]

  v2 = a1 + 408;
  sub_BBB200((__int64)v29);
  sub_983BD0((__int64)&v24, a1 + 176, a2);
  if ( *(_BYTE *)(a1 + 488) )
  {
    v3 = _mm_loadu_si128(&v25);
    v4 = _mm_loadu_si128(&v26);
    v5 = _mm_loadu_si128(&v27);
    v6 = _mm_loadu_si128(&v28);
    *(__m128i *)(a1 + 408) = _mm_loadu_si128(&v24);
    *(__m128i *)(a1 + 424) = v3;
    *(__m128i *)(a1 + 440) = v4;
    *(__m128i *)(a1 + 456) = v5;
    *(__m128i *)(a1 + 472) = v6;
  }
  else
  {
    v19 = _mm_loadu_si128(&v24);
    v20 = _mm_loadu_si128(&v25);
    *(_BYTE *)(a1 + 488) = 1;
    v21 = _mm_loadu_si128(&v26);
    v22 = _mm_loadu_si128(&v27);
    v23 = _mm_loadu_si128(&v28);
    *(__m128i *)(a1 + 408) = v19;
    *(__m128i *)(a1 + 424) = v20;
    *(__m128i *)(a1 + 440) = v21;
    *(__m128i *)(a1 + 456) = v22;
    *(__m128i *)(a1 + 472) = v23;
  }
  sub_C7D6A0(v34, 24LL * v35, 8);
  v7 = v33;
  if ( v33 )
  {
    v8 = (_QWORD **)(v32 + 32LL * v33);
    for ( i = (_QWORD **)(v32 + 8); ; i += 4 )
    {
      v10 = (__int64)*(i - 1);
      if ( v10 != -4096 && v10 != -8192 )
      {
        v11 = *i;
        while ( v11 != i )
        {
          v12 = (unsigned __int64)v11;
          v11 = (_QWORD *)*v11;
          v13 = *(_QWORD *)(v12 + 24);
          if ( v13 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
          j_j___libc_free_0(v12);
        }
      }
      if ( v8 == i + 3 )
        break;
    }
    v7 = v33;
  }
  sub_C7D6A0(v32, 32LL * v7, 8);
  v14 = v31;
  if ( v31 )
  {
    v15 = v30;
    v16 = &v30[2 * v31];
    do
    {
      if ( *v15 != -4096 && *v15 != -8192 )
      {
        v17 = v15[1];
        if ( v17 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
      }
      v15 += 2;
    }
    while ( v16 != v15 );
    v14 = v31;
  }
  sub_C7D6A0((__int64)v30, 16LL * v14, 8);
  return v2;
}
