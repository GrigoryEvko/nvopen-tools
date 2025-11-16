// Function: sub_2F3C110
// Address: 0x2f3c110
//
__int64 __fastcall sub_2F3C110(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r14
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  __m128i v9; // xmm3
  __m128i v10; // xmm4
  unsigned int v11; // eax
  _QWORD **v12; // r13
  _QWORD **i; // r12
  __int64 v14; // rax
  _QWORD *v15; // rbx
  unsigned __int64 v16; // r15
  __int64 v17; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  __int64 v21; // rdi
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  __m128i v25; // xmm7
  __m128i v26; // xmm0
  __m128i v27; // xmm1
  __m128i v28; // [rsp+0h] [rbp-E0h] BYREF
  __m128i v29; // [rsp+10h] [rbp-D0h] BYREF
  __m128i v30; // [rsp+20h] [rbp-C0h] BYREF
  __m128i v31; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v32; // [rsp+40h] [rbp-A0h] BYREF
  char v33[8]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD *v34; // [rsp+58h] [rbp-88h]
  unsigned int v35; // [rsp+68h] [rbp-78h]
  __int64 v36; // [rsp+78h] [rbp-68h]
  unsigned int v37; // [rsp+88h] [rbp-58h]
  __int64 v38; // [rsp+98h] [rbp-48h]
  unsigned int v39; // [rsp+A8h] [rbp-38h]

  v2 = *(__int64 **)(*(_QWORD *)a1 + 8LL);
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
  v6 = v5 + 408;
  sub_983BD0((__int64)&v28, v5 + 176, a2);
  if ( *(_BYTE *)(v5 + 488) )
  {
    v7 = _mm_loadu_si128(&v29);
    v8 = _mm_loadu_si128(&v30);
    v9 = _mm_loadu_si128(&v31);
    v10 = _mm_loadu_si128(&v32);
    *(__m128i *)(v5 + 408) = _mm_loadu_si128(&v28);
    *(__m128i *)(v5 + 424) = v7;
    *(__m128i *)(v5 + 440) = v8;
    *(__m128i *)(v5 + 456) = v9;
    *(__m128i *)(v5 + 472) = v10;
  }
  else
  {
    v23 = _mm_loadu_si128(&v28);
    v24 = _mm_loadu_si128(&v29);
    *(_BYTE *)(v5 + 488) = 1;
    v25 = _mm_loadu_si128(&v30);
    v26 = _mm_loadu_si128(&v31);
    v27 = _mm_loadu_si128(&v32);
    *(__m128i *)(v5 + 408) = v23;
    *(__m128i *)(v5 + 424) = v24;
    *(__m128i *)(v5 + 440) = v25;
    *(__m128i *)(v5 + 456) = v26;
    *(__m128i *)(v5 + 472) = v27;
  }
  sub_C7D6A0(v38, 24LL * v39, 8);
  v11 = v37;
  if ( v37 )
  {
    v12 = (_QWORD **)(v36 + 32LL * v37);
    for ( i = (_QWORD **)(v36 + 8); ; i += 4 )
    {
      v14 = (__int64)*(i - 1);
      if ( v14 != -8192 && v14 != -4096 )
      {
        v15 = *i;
        while ( v15 != i )
        {
          v16 = (unsigned __int64)v15;
          v15 = (_QWORD *)*v15;
          v17 = *(_QWORD *)(v16 + 24);
          if ( v17 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
          j_j___libc_free_0(v16);
        }
      }
      if ( v12 == i + 3 )
        break;
    }
    v11 = v37;
  }
  sub_C7D6A0(v36, 32LL * v11, 8);
  v18 = v35;
  if ( v35 )
  {
    v19 = v34;
    v20 = &v34[2 * v35];
    do
    {
      if ( *v19 != -8192 && *v19 != -4096 )
      {
        v21 = v19[1];
        if ( v21 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
      }
      v19 += 2;
    }
    while ( v20 != v19 );
    v18 = v35;
  }
  sub_C7D6A0((__int64)v34, 16LL * v18, 8);
  return v6;
}
