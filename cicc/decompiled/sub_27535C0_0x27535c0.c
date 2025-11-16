// Function: sub_27535C0
// Address: 0x27535c0
//
__int64 __fastcall sub_27535C0(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __m128i v7; // xmm5
  __m128i v8; // xmm6
  __m128i v9; // xmm7
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  unsigned int v12; // eax
  _QWORD **v13; // r13
  _QWORD **i; // r12
  __int64 v15; // rax
  _QWORD *v16; // rbx
  unsigned __int64 v17; // r15
  __int64 v18; // rdi
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  unsigned int v23; // eax
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  __int64 v26; // rdi
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

  if ( (unsigned __int8)sub_BB98D0(a1, a2) )
    return 0;
  v3 = (__int64 *)a1[1];
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4F6D3F0 )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_30;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v33);
  sub_983BD0((__int64)&v28, v6 + 176, a2);
  v27 = (__int64 *)(v6 + 408);
  if ( *(_BYTE *)(v6 + 488) )
  {
    v19 = _mm_loadu_si128(&v29);
    v20 = _mm_loadu_si128(&v30);
    v21 = _mm_loadu_si128(&v31);
    v22 = _mm_loadu_si128(&v32);
    *(__m128i *)(v6 + 408) = _mm_loadu_si128(&v28);
    *(__m128i *)(v6 + 424) = v19;
    *(__m128i *)(v6 + 440) = v20;
    *(__m128i *)(v6 + 456) = v21;
    *(__m128i *)(v6 + 472) = v22;
  }
  else
  {
    v7 = _mm_loadu_si128(&v28);
    v8 = _mm_loadu_si128(&v29);
    *(_BYTE *)(v6 + 488) = 1;
    v9 = _mm_loadu_si128(&v30);
    v10 = _mm_loadu_si128(&v31);
    v11 = _mm_loadu_si128(&v32);
    *(__m128i *)(v6 + 408) = v7;
    *(__m128i *)(v6 + 424) = v8;
    *(__m128i *)(v6 + 440) = v9;
    *(__m128i *)(v6 + 456) = v10;
    *(__m128i *)(v6 + 472) = v11;
  }
  sub_C7D6A0(v38, 24LL * v39, 8);
  v12 = v37;
  if ( v37 )
  {
    v13 = (_QWORD **)(v36 + 32LL * v37);
    for ( i = (_QWORD **)(v36 + 8); ; i += 4 )
    {
      v15 = (__int64)*(i - 1);
      if ( v15 != -8192 && v15 != -4096 )
      {
        v16 = *i;
        while ( v16 != i )
        {
          v17 = (unsigned __int64)v16;
          v16 = (_QWORD *)*v16;
          v18 = *(_QWORD *)(v17 + 24);
          if ( v18 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
          j_j___libc_free_0(v17);
        }
      }
      if ( v13 == i + 3 )
        break;
    }
    v12 = v37;
  }
  sub_C7D6A0(v36, 32LL * v12, 8);
  v23 = v35;
  if ( v35 )
  {
    v24 = v34;
    v25 = &v34[2 * v35];
    do
    {
      if ( *v24 != -8192 && *v24 != -4096 )
      {
        v26 = v24[1];
        if ( v26 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
      }
      v24 += 2;
    }
    while ( v25 != v24 );
    v23 = v35;
  }
  sub_C7D6A0((__int64)v34, 16LL * v23, 8);
  return sub_2753060(a2, v27);
}
