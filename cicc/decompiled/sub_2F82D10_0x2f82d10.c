// Function: sub_2F82D10
// Address: 0x2f82d10
//
__int64 __fastcall sub_2F82D10(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  __m128i v9; // xmm4
  unsigned int v10; // eax
  _QWORD **v11; // r13
  _QWORD **i; // r12
  __int64 v13; // rax
  _QWORD *v14; // rbx
  unsigned __int64 v15; // r15
  __int64 v16; // rdi
  unsigned int v17; // eax
  _QWORD *v18; // rbx
  _QWORD *v19; // r12
  __int64 v20; // rdi
  __m128i v22; // xmm5
  __m128i v23; // xmm6
  __m128i v24; // xmm7
  __m128i v25; // xmm0
  __m128i v26; // xmm1
  __int64 *v27; // [rsp+8h] [rbp-E8h]
  __m128i v28; // [rsp+10h] [rbp-E0h] BYREF
  __m128i v29; // [rsp+20h] [rbp-D0h] BYREF
  __m128i v30; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v31; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v32; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE v33[8]; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v34; // [rsp+68h] [rbp-88h]
  unsigned int v35; // [rsp+78h] [rbp-78h]
  __int64 v36; // [rsp+88h] [rbp-68h]
  unsigned int v37; // [rsp+98h] [rbp-58h]
  __int64 v38; // [rsp+A8h] [rbp-48h]
  unsigned int v39; // [rsp+B8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F6D3F0 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_28;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v33);
  sub_983BD0((__int64)&v28, v5 + 176, a2);
  v27 = (__int64 *)(v5 + 408);
  if ( *(_BYTE *)(v5 + 488) )
  {
    v6 = _mm_loadu_si128(&v29);
    v7 = _mm_loadu_si128(&v30);
    v8 = _mm_loadu_si128(&v31);
    v9 = _mm_loadu_si128(&v32);
    *(__m128i *)(v5 + 408) = _mm_loadu_si128(&v28);
    *(__m128i *)(v5 + 424) = v6;
    *(__m128i *)(v5 + 440) = v7;
    *(__m128i *)(v5 + 456) = v8;
    *(__m128i *)(v5 + 472) = v9;
  }
  else
  {
    v22 = _mm_loadu_si128(&v28);
    v23 = _mm_loadu_si128(&v29);
    *(_BYTE *)(v5 + 488) = 1;
    v24 = _mm_loadu_si128(&v30);
    v25 = _mm_loadu_si128(&v31);
    v26 = _mm_loadu_si128(&v32);
    *(__m128i *)(v5 + 408) = v22;
    *(__m128i *)(v5 + 424) = v23;
    *(__m128i *)(v5 + 440) = v24;
    *(__m128i *)(v5 + 456) = v25;
    *(__m128i *)(v5 + 472) = v26;
  }
  sub_C7D6A0(v38, 24LL * v39, 8);
  v10 = v37;
  if ( v37 )
  {
    v11 = (_QWORD **)(v36 + 32LL * v37);
    for ( i = (_QWORD **)(v36 + 8); ; i += 4 )
    {
      v13 = (__int64)*(i - 1);
      if ( v13 != -8192 && v13 != -4096 )
      {
        v14 = *i;
        while ( v14 != i )
        {
          v15 = (unsigned __int64)v14;
          v14 = (_QWORD *)*v14;
          v16 = *(_QWORD *)(v15 + 24);
          if ( v16 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
          j_j___libc_free_0(v15);
        }
      }
      if ( v11 == i + 3 )
        break;
    }
    v10 = v37;
  }
  sub_C7D6A0(v36, 32LL * v10, 8);
  v17 = v35;
  if ( v35 )
  {
    v18 = v34;
    v19 = &v34[2 * v35];
    do
    {
      if ( *v18 != -4096 && *v18 != -8192 )
      {
        v20 = v18[1];
        if ( v20 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
      }
      v18 += 2;
    }
    while ( v19 != v18 );
    v17 = v35;
  }
  sub_C7D6A0((__int64)v34, 16LL * v17, 8);
  return sub_2F81900(v27, a2);
}
