// Function: sub_FFA3F0
// Address: 0xffa3f0
//
__int64 __fastcall sub_FFA3F0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // xmm4
  unsigned int v15; // eax
  _QWORD **v16; // r12
  _QWORD **i; // rbx
  __int64 v18; // rax
  _QWORD *v19; // r15
  _QWORD *v20; // r14
  __int64 v21; // rdi
  unsigned int v22; // eax
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  __int64 v25; // rdi
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __m128i v36; // xmm5
  __m128i v37; // xmm6
  __m128i v38; // xmm7
  __m128i v39; // xmm0
  __m128i v40; // xmm1
  __int64 v41; // [rsp+8h] [rbp-F8h]
  __int64 *v42; // [rsp+10h] [rbp-F0h]
  __m128i v43; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v44; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v45; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v46; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v47; // [rsp+60h] [rbp-A0h] BYREF
  _BYTE v48[8]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v49; // [rsp+78h] [rbp-88h]
  unsigned int v50; // [rsp+88h] [rbp-78h]
  __int64 v51; // [rsp+98h] [rbp-68h]
  unsigned int v52; // [rsp+A8h] [rbp-58h]
  __int64 v53; // [rsp+B8h] [rbp-48h]
  unsigned int v54; // [rsp+C8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_49:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F875EC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_49;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F875EC);
  v7 = *(__int64 **)(a1 + 8);
  v41 = v6 + 176;
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_46:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F6D3F0 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_46;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v48);
  sub_983BD0((__int64)&v43, v10 + 176, a2);
  v42 = (__int64 *)(v10 + 408);
  if ( *(_BYTE *)(v10 + 488) )
  {
    v11 = _mm_loadu_si128(&v44);
    v12 = _mm_loadu_si128(&v45);
    v13 = _mm_loadu_si128(&v46);
    v14 = _mm_loadu_si128(&v47);
    *(__m128i *)(v10 + 408) = _mm_loadu_si128(&v43);
    *(__m128i *)(v10 + 424) = v11;
    *(__m128i *)(v10 + 440) = v12;
    *(__m128i *)(v10 + 456) = v13;
    *(__m128i *)(v10 + 472) = v14;
  }
  else
  {
    v36 = _mm_loadu_si128(&v43);
    v37 = _mm_loadu_si128(&v44);
    *(_BYTE *)(v10 + 488) = 1;
    v38 = _mm_loadu_si128(&v45);
    v39 = _mm_loadu_si128(&v46);
    v40 = _mm_loadu_si128(&v47);
    *(__m128i *)(v10 + 408) = v36;
    *(__m128i *)(v10 + 424) = v37;
    *(__m128i *)(v10 + 440) = v38;
    *(__m128i *)(v10 + 456) = v39;
    *(__m128i *)(v10 + 472) = v40;
  }
  sub_C7D6A0(v53, 24LL * v54, 8);
  v15 = v52;
  if ( v52 )
  {
    v16 = (_QWORD **)(v51 + 32LL * v52);
    for ( i = (_QWORD **)(v51 + 8); ; i += 4 )
    {
      v18 = (__int64)*(i - 1);
      if ( v18 != -8192 && v18 != -4096 )
      {
        v19 = *i;
        while ( v19 != i )
        {
          v20 = v19;
          v19 = (_QWORD *)*v19;
          v21 = v20[3];
          if ( v21 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
          j_j___libc_free_0(v20, 32);
        }
      }
      if ( v16 == i + 3 )
        break;
    }
    v15 = v52;
  }
  sub_C7D6A0(v51, 32LL * v15, 8);
  v22 = v50;
  if ( v50 )
  {
    v23 = v49;
    v24 = &v49[2 * v50];
    do
    {
      if ( *v23 != -8192 && *v23 != -4096 )
      {
        v25 = v23[1];
        if ( v25 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
      }
      v23 += 2;
    }
    while ( v24 != v23 );
    v22 = v50;
  }
  sub_C7D6A0((__int64)v49, 16LL * v22, 8);
  v26 = *(__int64 **)(a1 + 8);
  v27 = *v26;
  v28 = v26[1];
  if ( v27 == v28 )
LABEL_47:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4F8144C )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_47;
  }
  v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(*(_QWORD *)(v27 + 8), &unk_4F8144C);
  v30 = *(__int64 **)(a1 + 8);
  v31 = v29 + 176;
  v32 = *v30;
  v33 = v30[1];
  if ( v32 == v33 )
LABEL_48:
    BUG();
  while ( *(_UNKNOWN **)v32 != &unk_4F8FBD4 )
  {
    v32 += 16;
    if ( v33 == v32 )
      goto LABEL_48;
  }
  v34 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v32 + 8) + 104LL))(*(_QWORD *)(v32 + 8), &unk_4F8FBD4);
  sub_FF9360((_QWORD *)(a1 + 176), a2, v41, v42, v31, v34 + 176);
  return 0;
}
