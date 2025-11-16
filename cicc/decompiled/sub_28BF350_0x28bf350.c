// Function: sub_28BF350
// Address: 0x28bf350
//
__int64 __fastcall sub_28BF350(_QWORD *a1, __int64 a2)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  __m128i v8; // xmm5
  __m128i v9; // xmm6
  __m128i v10; // xmm7
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  unsigned int v13; // eax
  _QWORD **v14; // rbx
  __int64 v15; // rax
  _QWORD *v16; // r15
  unsigned __int64 v17; // r12
  __int64 v18; // rdi
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  unsigned int v23; // eax
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  __int64 v26; // rdi
  __int64 *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 *v31; // r12
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 *v40; // rdx
  __int64 *v41; // [rsp+0h] [rbp-F0h]
  _QWORD **v42; // [rsp+8h] [rbp-E8h]
  __m128i v43; // [rsp+10h] [rbp-E0h] BYREF
  __m128i v44; // [rsp+20h] [rbp-D0h] BYREF
  __m128i v45; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v46; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v47; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE v48[8]; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v49; // [rsp+68h] [rbp-88h]
  unsigned int v50; // [rsp+78h] [rbp-78h]
  __int64 v51; // [rsp+88h] [rbp-68h]
  unsigned int v52; // [rsp+98h] [rbp-58h]
  __int64 v53; // [rsp+A8h] [rbp-48h]
  unsigned int v54; // [rsp+B8h] [rbp-38h]

  if ( (unsigned __int8)sub_BB98D0(a1, a2) )
    return 0;
  v4 = (__int64 *)a1[1];
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
    goto LABEL_43;
  while ( *(_UNKNOWN **)v5 != &unk_4F6D3F0 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_43;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v48);
  sub_983BD0((__int64)&v43, v7 + 176, a2);
  v41 = (__int64 *)(v7 + 408);
  if ( *(_BYTE *)(v7 + 488) )
  {
    v19 = _mm_loadu_si128(&v44);
    v20 = _mm_loadu_si128(&v45);
    v21 = _mm_loadu_si128(&v46);
    v22 = _mm_loadu_si128(&v47);
    *(__m128i *)(v7 + 408) = _mm_loadu_si128(&v43);
    *(__m128i *)(v7 + 424) = v19;
    *(__m128i *)(v7 + 440) = v20;
    *(__m128i *)(v7 + 456) = v21;
    *(__m128i *)(v7 + 472) = v22;
  }
  else
  {
    v8 = _mm_loadu_si128(&v43);
    v9 = _mm_loadu_si128(&v44);
    *(_BYTE *)(v7 + 488) = 1;
    v10 = _mm_loadu_si128(&v45);
    v11 = _mm_loadu_si128(&v46);
    v12 = _mm_loadu_si128(&v47);
    *(__m128i *)(v7 + 408) = v8;
    *(__m128i *)(v7 + 424) = v9;
    *(__m128i *)(v7 + 440) = v10;
    *(__m128i *)(v7 + 456) = v11;
    *(__m128i *)(v7 + 472) = v12;
  }
  sub_C7D6A0(v53, 24LL * v54, 8);
  v13 = v52;
  if ( v52 )
  {
    v14 = (_QWORD **)(v51 + 8);
    v42 = (_QWORD **)(v51 + 32LL * v52);
    while ( 1 )
    {
      v15 = (__int64)*(v14 - 1);
      if ( v15 != -8192 && v15 != -4096 )
      {
        v16 = *v14;
        while ( v16 != v14 )
        {
          v17 = (unsigned __int64)v16;
          v16 = (_QWORD *)*v16;
          v18 = *(_QWORD *)(v17 + 24);
          if ( v18 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
          j_j___libc_free_0(v17);
        }
      }
      if ( v42 == v14 + 3 )
        break;
      v14 += 4;
    }
    v13 = v52;
  }
  sub_C7D6A0(v51, 32LL * v13, 8);
  v23 = v50;
  if ( v50 )
  {
    v24 = v49;
    v25 = &v49[2 * v50];
    do
    {
      if ( *v24 != -4096 && *v24 != -8192 )
      {
        v26 = v24[1];
        if ( v26 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
      }
      v24 += 2;
    }
    while ( v25 != v24 );
    v23 = v50;
  }
  sub_C7D6A0((__int64)v49, 16LL * v23, 8);
  v27 = (__int64 *)a1[1];
  v28 = *v27;
  v29 = v27[1];
  if ( v28 == v29 )
LABEL_43:
    BUG();
  while ( *(_UNKNOWN **)v28 != &unk_4F89C28 )
  {
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_43;
  }
  v30 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(*(_QWORD *)(v28 + 8), &unk_4F89C28);
  v31 = (__int64 *)sub_DFED00(v30, a2);
  v32 = sub_B82360(a1[1], (__int64)&unk_4F8144C);
  v33 = v32;
  if ( v32 )
  {
    v34 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v32 + 104LL))(v32, &unk_4F8144C);
    v35 = (__int64 *)a1[1];
    v33 = v34;
    v36 = *v35;
    v37 = v35[1];
    if ( v36 != v37 )
      goto LABEL_37;
    goto LABEL_43;
  }
  v40 = (__int64 *)a1[1];
  v36 = *v40;
  v37 = v40[1];
  if ( v36 == v37 )
    BUG();
LABEL_37:
  while ( *(_UNKNOWN **)v36 != &unk_4F86530 )
  {
    v36 += 16;
    if ( v37 == v36 )
      goto LABEL_43;
  }
  v38 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v36 + 8) + 104LL))(*(_QWORD *)(v36 + 8), &unk_4F86530);
  v39 = v33 + 176;
  if ( !v33 )
    v39 = 0;
  return sub_28BF280(a2, v41, v31, *(_QWORD *)(v38 + 176), v39);
}
