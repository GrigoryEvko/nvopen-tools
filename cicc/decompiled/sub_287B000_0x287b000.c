// Function: sub_287B000
// Address: 0x287b000
//
__int64 __fastcall sub_287B000(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
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
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rbx
  __m128i v34; // xmm5
  __m128i v35; // xmm6
  __m128i v36; // xmm7
  __m128i v37; // xmm0
  __m128i v38; // xmm1
  unsigned int v39; // eax
  _QWORD **v40; // r15
  _QWORD **i; // rbx
  __int64 v42; // rax
  _QWORD *v43; // r12
  unsigned __int64 v44; // r13
  __int64 v45; // rdi
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  __m128i v48; // xmm3
  __m128i v49; // xmm4
  unsigned int v50; // eax
  _QWORD *v51; // rbx
  _QWORD *v52; // r12
  __int64 v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rax
  __int64 v58; // [rsp+8h] [rbp-118h]
  __int64 v59; // [rsp+10h] [rbp-110h]
  __int64 *v60; // [rsp+10h] [rbp-110h]
  __int64 v61; // [rsp+18h] [rbp-108h]
  __int64 v62; // [rsp+20h] [rbp-100h]
  __int64 v63; // [rsp+28h] [rbp-F8h]
  __int64 v64; // [rsp+30h] [rbp-F0h]
  __int64 v65; // [rsp+38h] [rbp-E8h]
  __m128i v66; // [rsp+40h] [rbp-E0h] BYREF
  __m128i v67; // [rsp+50h] [rbp-D0h] BYREF
  __m128i v68; // [rsp+60h] [rbp-C0h] BYREF
  __m128i v69; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v70; // [rsp+80h] [rbp-A0h] BYREF
  _BYTE v71[8]; // [rsp+90h] [rbp-90h] BYREF
  _QWORD *v72; // [rsp+98h] [rbp-88h]
  unsigned int v73; // [rsp+A8h] [rbp-78h]
  __int64 v74; // [rsp+B8h] [rbp-68h]
  unsigned int v75; // [rsp+C8h] [rbp-58h]
  __int64 v76; // [rsp+D8h] [rbp-48h]
  unsigned int v77; // [rsp+E8h] [rbp-38h]

  v2 = a1;
  if ( (unsigned __int8)sub_D58140(a1, a2) )
    return 0;
  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_71:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4FDB6AC )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_71;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4FDB6AC);
  v8 = *(__int64 **)(a1 + 8);
  v58 = *(_QWORD *)(v7 + 176);
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
LABEL_73:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F881C8 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_73;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F881C8);
  v12 = *(__int64 **)(a1 + 8);
  v65 = *(_QWORD *)(v11 + 176);
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
LABEL_74:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F8144C )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_74;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F8144C);
  v16 = *(__int64 **)(a1 + 8);
  v64 = v15 + 176;
  v17 = *v16;
  v18 = v16[1];
  if ( v17 == v18 )
LABEL_75:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F875EC )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_75;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F875EC);
  v20 = *(__int64 **)(a1 + 8);
  v63 = v19 + 176;
  v21 = *v20;
  v22 = v20[1];
  if ( v21 == v22 )
LABEL_76:
    BUG();
  while ( *(_UNKNOWN **)v21 != &unk_4F89C28 )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_76;
  }
  v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(*(_QWORD *)(v21 + 8), &unk_4F89C28);
  v24 = sub_DFED00(v23, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL));
  v25 = *(__int64 **)(a1 + 8);
  v62 = v24;
  v26 = *v25;
  v27 = v25[1];
  if ( v26 == v27 )
LABEL_77:
    BUG();
  while ( *(_UNKNOWN **)v26 != &unk_4F8662C )
  {
    v26 += 16;
    if ( v27 == v26 )
      goto LABEL_77;
  }
  v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(*(_QWORD *)(v26 + 8), &unk_4F8662C);
  v29 = sub_CFFAC0(v28, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL));
  v30 = *(__int64 **)(a1 + 8);
  v61 = v29;
  v31 = *v30;
  v32 = v30[1];
  if ( v31 == v32 )
LABEL_72:
    BUG();
  while ( *(_UNKNOWN **)v31 != &unk_4F6D3F0 )
  {
    v31 += 16;
    if ( v32 == v31 )
      goto LABEL_72;
  }
  v33 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(*(_QWORD *)(v31 + 8), &unk_4F6D3F0);
  v59 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL);
  sub_BBB200((__int64)v71);
  sub_983BD0((__int64)&v66, v33 + 176, v59);
  v60 = (__int64 *)(v33 + 408);
  if ( *(_BYTE *)(v33 + 488) )
  {
    v46 = _mm_loadu_si128(&v67);
    v47 = _mm_loadu_si128(&v68);
    v48 = _mm_loadu_si128(&v69);
    v49 = _mm_loadu_si128(&v70);
    *(__m128i *)(v33 + 408) = _mm_loadu_si128(&v66);
    *(__m128i *)(v33 + 424) = v46;
    *(__m128i *)(v33 + 440) = v47;
    *(__m128i *)(v33 + 456) = v48;
    *(__m128i *)(v33 + 472) = v49;
  }
  else
  {
    v34 = _mm_loadu_si128(&v66);
    v35 = _mm_loadu_si128(&v67);
    *(_BYTE *)(v33 + 488) = 1;
    v36 = _mm_loadu_si128(&v68);
    v37 = _mm_loadu_si128(&v69);
    v38 = _mm_loadu_si128(&v70);
    *(__m128i *)(v33 + 408) = v34;
    *(__m128i *)(v33 + 424) = v35;
    *(__m128i *)(v33 + 440) = v36;
    *(__m128i *)(v33 + 456) = v37;
    *(__m128i *)(v33 + 472) = v38;
  }
  sub_C7D6A0(v76, 24LL * v77, 8);
  v39 = v75;
  if ( v75 )
  {
    v40 = (_QWORD **)(v74 + 32LL * v75);
    for ( i = (_QWORD **)(v74 + 8); ; i += 4 )
    {
      v42 = (__int64)*(i - 1);
      if ( v42 != -8192 && v42 != -4096 )
      {
        v43 = *i;
        while ( v43 != i )
        {
          v44 = (unsigned __int64)v43;
          v43 = (_QWORD *)*v43;
          v45 = *(_QWORD *)(v44 + 24);
          if ( v45 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v45 + 8LL))(v45);
          j_j___libc_free_0(v44);
        }
      }
      if ( v40 == i + 3 )
        break;
    }
    v2 = a1;
    v39 = v75;
  }
  sub_C7D6A0(v74, 32LL * v39, 8);
  v50 = v73;
  if ( v73 )
  {
    v51 = v72;
    v52 = &v72[2 * v73];
    do
    {
      if ( *v51 != -8192 && *v51 != -4096 )
      {
        v53 = v51[1];
        if ( v53 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v53 + 8LL))(v53);
      }
      v51 += 2;
    }
    while ( v52 != v51 );
    v50 = v73;
  }
  sub_C7D6A0((__int64)v72, 16LL * v50, 8);
  v54 = sub_B82360(*(_QWORD *)(v2 + 8), (__int64)&unk_4F8F808);
  if ( v54 && (v55 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v54 + 104LL))(v54, &unk_4F8F808)) != 0 )
    v56 = *(_QWORD *)(v55 + 176);
  else
    v56 = 0;
  return sub_2877B80(a2, v58, v65, v64, v63, v62, v61, v60, v56);
}
