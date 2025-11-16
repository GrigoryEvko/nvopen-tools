// Function: sub_287CD90
// Address: 0x287cd90
//
__int64 __fastcall sub_287CD90(__int64 a1, __int64 a2)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // r15
  __m128i v25; // xmm5
  __m128i v26; // xmm6
  __m128i v27; // xmm7
  __m128i v28; // xmm0
  __m128i v29; // xmm1
  unsigned int v30; // eax
  _QWORD **v31; // rbx
  __int64 v32; // rax
  _QWORD *v33; // r15
  unsigned __int64 v34; // r12
  __int64 v35; // rdi
  __m128i v36; // xmm1
  __m128i v37; // xmm2
  __m128i v38; // xmm3
  __m128i v39; // xmm4
  unsigned int v40; // eax
  _QWORD *v41; // rbx
  _QWORD *v42; // r12
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r9
  __int64 *v47; // [rsp+8h] [rbp-108h]
  unsigned __int64 v48; // [rsp+10h] [rbp-100h]
  __int64 v49; // [rsp+18h] [rbp-F8h]
  __int64 v50; // [rsp+20h] [rbp-F0h]
  _QWORD **v51; // [rsp+28h] [rbp-E8h]
  __m128i v52; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v53; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v54; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v55; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v56; // [rsp+70h] [rbp-A0h] BYREF
  _BYTE v57[8]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD *v58; // [rsp+88h] [rbp-88h]
  unsigned int v59; // [rsp+98h] [rbp-78h]
  __int64 v60; // [rsp+A8h] [rbp-68h]
  unsigned int v61; // [rsp+B8h] [rbp-58h]
  __int64 v62; // [rsp+C8h] [rbp-48h]
  unsigned int v63; // [rsp+D8h] [rbp-38h]

  if ( (unsigned __int8)sub_D58140(a1, a2) )
    return 0;
  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
    goto LABEL_50;
  while ( *(_UNKNOWN **)v5 != &unk_4F881C8 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_50;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F881C8);
  v8 = *(__int64 **)(a1 + 8);
  v47 = *(__int64 **)(v7 + 176);
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
    goto LABEL_50;
  while ( *(_UNKNOWN **)v9 != &unk_4F8144C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_50;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F8144C);
  v12 = *(__int64 **)(a1 + 8);
  v50 = v11 + 176;
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
    goto LABEL_50;
  while ( *(_UNKNOWN **)v13 != &unk_4F875EC )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_50;
  }
  (*(void (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F875EC);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *v15;
  v17 = v15[1];
  if ( v16 == v17 )
    goto LABEL_50;
  while ( *(_UNKNOWN **)v16 != &unk_4F89C28 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_50;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F89C28);
  v19 = sub_DFED00(v18, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL));
  v20 = *(__int64 **)(a1 + 8);
  v49 = v19;
  v21 = *v20;
  v22 = v20[1];
  if ( v21 == v22 )
LABEL_50:
    BUG();
  while ( *(_UNKNOWN **)v21 != &unk_4F6D3F0 )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_50;
  }
  v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(*(_QWORD *)(v21 + 8), &unk_4F6D3F0);
  v24 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL);
  sub_BBB200((__int64)v57);
  sub_983BD0((__int64)&v52, v23 + 176, v24);
  v48 = v23 + 408;
  if ( *(_BYTE *)(v23 + 488) )
  {
    v36 = _mm_loadu_si128(&v53);
    v37 = _mm_loadu_si128(&v54);
    v38 = _mm_loadu_si128(&v55);
    v39 = _mm_loadu_si128(&v56);
    *(__m128i *)(v23 + 408) = _mm_loadu_si128(&v52);
    *(__m128i *)(v23 + 424) = v36;
    *(__m128i *)(v23 + 440) = v37;
    *(__m128i *)(v23 + 456) = v38;
    *(__m128i *)(v23 + 472) = v39;
  }
  else
  {
    v25 = _mm_loadu_si128(&v52);
    v26 = _mm_loadu_si128(&v53);
    *(_BYTE *)(v23 + 488) = 1;
    v27 = _mm_loadu_si128(&v54);
    v28 = _mm_loadu_si128(&v55);
    v29 = _mm_loadu_si128(&v56);
    *(__m128i *)(v23 + 408) = v25;
    *(__m128i *)(v23 + 424) = v26;
    *(__m128i *)(v23 + 440) = v27;
    *(__m128i *)(v23 + 456) = v28;
    *(__m128i *)(v23 + 472) = v29;
  }
  sub_C7D6A0(v62, 24LL * v63, 8);
  v30 = v61;
  if ( v61 )
  {
    v31 = (_QWORD **)(v60 + 8);
    v51 = (_QWORD **)(v60 + 32LL * v61);
    while ( 1 )
    {
      v32 = (__int64)*(v31 - 1);
      if ( v32 != -8192 && v32 != -4096 )
      {
        v33 = *v31;
        while ( v33 != v31 )
        {
          v34 = (unsigned __int64)v33;
          v33 = (_QWORD *)*v33;
          v35 = *(_QWORD *)(v34 + 24);
          if ( v35 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v35 + 8LL))(v35);
          j_j___libc_free_0(v34);
        }
      }
      if ( v51 == v31 + 3 )
        break;
      v31 += 4;
    }
    v30 = v61;
  }
  sub_C7D6A0(v60, 32LL * v30, 8);
  v40 = v59;
  if ( v59 )
  {
    v41 = v58;
    v42 = &v58[2 * v59];
    do
    {
      if ( *v41 != -8192 && *v41 != -4096 )
      {
        v43 = v41[1];
        if ( v43 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v43 + 8LL))(v43);
      }
      v41 += 2;
    }
    while ( v42 != v41 );
    v40 = v59;
  }
  sub_C7D6A0((__int64)v58, 16LL * v40, 8);
  v44 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8F808);
  if ( v44 && (v45 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v44 + 104LL))(v44, &unk_4F8F808)) != 0 )
    v46 = *(_QWORD *)(v45 + 176);
  else
    v46 = 0;
  return sub_287C150(a2, v47, v50, v49, v48, v46);
}
