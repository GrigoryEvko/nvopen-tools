// Function: sub_27FE1C0
// Address: 0x27fe1c0
//
__int64 __fastcall sub_27FE1C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  __m128i v18; // xmm5
  __m128i v19; // xmm6
  __m128i v20; // xmm7
  __m128i v21; // xmm0
  __m128i v22; // xmm1
  unsigned int v23; // eax
  _QWORD **v24; // r15
  _QWORD **i; // r14
  __int64 v26; // rax
  _QWORD *v27; // r12
  unsigned __int64 v28; // r13
  __int64 v29; // rdi
  __m128i v30; // xmm1
  __m128i v31; // xmm2
  __m128i v32; // xmm3
  __m128i v33; // xmm4
  unsigned int v34; // eax
  _QWORD *v35; // r14
  _QWORD *v36; // r15
  __int64 v37; // rdi
  __int64 *v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 *v43; // rdx
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 *v48; // rdx
  __int64 *v49; // r14
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 *v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rax
  __int64 v56; // rdx
  _QWORD *v57; // rbx
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 *v62; // r13
  __int64 **v63; // [rsp+10h] [rbp-140h]
  __int64 v64; // [rsp+18h] [rbp-138h]
  __int64 v65; // [rsp+20h] [rbp-130h]
  __int64 *v66; // [rsp+28h] [rbp-128h]
  __int64 v67; // [rsp+38h] [rbp-118h]
  __int64 v68; // [rsp+38h] [rbp-118h]
  __int64 *v69; // [rsp+40h] [rbp-110h]
  unsigned __int8 v70; // [rsp+4Fh] [rbp-101h]
  __int64 v71[2]; // [rsp+50h] [rbp-100h] BYREF
  __int64 *v72; // [rsp+60h] [rbp-F0h]
  __m128i v73; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v74; // [rsp+80h] [rbp-D0h] BYREF
  __m128i v75; // [rsp+90h] [rbp-C0h] BYREF
  __m128i v76; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v77; // [rsp+B0h] [rbp-A0h] BYREF
  _BYTE v78[8]; // [rsp+C0h] [rbp-90h] BYREF
  _QWORD *v79; // [rsp+C8h] [rbp-88h]
  unsigned int v80; // [rsp+D8h] [rbp-78h]
  __int64 v81; // [rsp+E8h] [rbp-68h]
  unsigned int v82; // [rsp+F8h] [rbp-58h]
  __int64 v83; // [rsp+108h] [rbp-48h]
  unsigned int v84; // [rsp+118h] [rbp-38h]

  v2 = a2;
  v70 = sub_D58140(a1, a2);
  if ( v70 )
  {
    return 0;
  }
  else
  {
    v67 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL);
    v5 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F881C8);
    v69 = (__int64 *)v5;
    if ( v5 )
      v69 = (__int64 *)(*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v5 + 104LL))(v5, &unk_4F881C8);
    v6 = *(__int64 **)(a1 + 8);
    v7 = *v6;
    v8 = v6[1];
    if ( v7 == v8 )
LABEL_73:
      BUG();
    while ( *(_UNKNOWN **)v7 != &unk_4F8F808 )
    {
      v7 += 16;
      if ( v8 == v7 )
        goto LABEL_73;
    }
    v65 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(
                        *(_QWORD *)(v7 + 8),
                        &unk_4F8F808)
                    + 176);
    sub_1049690(v71, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL));
    v64 = a1 + 172;
    if ( v69 )
      v69 = (__int64 *)v69[22];
    v9 = *(__int64 **)(a1 + 8);
    v10 = *v9;
    v11 = v9[1];
    if ( v10 == v11 )
LABEL_74:
      BUG();
    while ( *(_UNKNOWN **)v10 != &unk_4F89C28 )
    {
      v10 += 16;
      if ( v11 == v10 )
        goto LABEL_74;
    }
    v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(
            *(_QWORD *)(v10 + 8),
            &unk_4F89C28);
    v13 = sub_DFED00(v12, v67);
    v14 = *(__int64 **)(a1 + 8);
    v63 = (__int64 **)v13;
    v15 = *v14;
    v16 = v14[1];
    if ( v15 == v16 )
LABEL_75:
      BUG();
    while ( *(_UNKNOWN **)v15 != &unk_4F6D3F0 )
    {
      v15 += 16;
      if ( v16 == v15 )
        goto LABEL_75;
    }
    v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(
            *(_QWORD *)(v15 + 8),
            &unk_4F6D3F0);
    sub_BBB200((__int64)v78);
    sub_983BD0((__int64)&v73, v17 + 176, v67);
    v66 = (__int64 *)(v17 + 408);
    if ( *(_BYTE *)(v17 + 488) )
    {
      v30 = _mm_loadu_si128(&v74);
      v31 = _mm_loadu_si128(&v75);
      v32 = _mm_loadu_si128(&v76);
      v33 = _mm_loadu_si128(&v77);
      *(__m128i *)(v17 + 408) = _mm_loadu_si128(&v73);
      *(__m128i *)(v17 + 424) = v30;
      *(__m128i *)(v17 + 440) = v31;
      *(__m128i *)(v17 + 456) = v32;
      *(__m128i *)(v17 + 472) = v33;
    }
    else
    {
      v18 = _mm_loadu_si128(&v73);
      v19 = _mm_loadu_si128(&v74);
      *(_BYTE *)(v17 + 488) = 1;
      v20 = _mm_loadu_si128(&v75);
      v21 = _mm_loadu_si128(&v76);
      v22 = _mm_loadu_si128(&v77);
      *(__m128i *)(v17 + 408) = v18;
      *(__m128i *)(v17 + 424) = v19;
      *(__m128i *)(v17 + 440) = v20;
      *(__m128i *)(v17 + 456) = v21;
      *(__m128i *)(v17 + 472) = v22;
    }
    sub_C7D6A0(v83, 24LL * v84, 8);
    v23 = v82;
    if ( v82 )
    {
      v24 = (_QWORD **)(v81 + 32LL * v82);
      for ( i = (_QWORD **)(v81 + 8); ; i += 4 )
      {
        v26 = (__int64)*(i - 1);
        if ( v26 != -4096 && v26 != -8192 )
        {
          v27 = *i;
          while ( v27 != i )
          {
            v28 = (unsigned __int64)v27;
            v27 = (_QWORD *)*v27;
            v29 = *(_QWORD *)(v28 + 24);
            if ( v29 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v29 + 8LL))(v29);
            j_j___libc_free_0(v28);
          }
        }
        if ( v24 == i + 3 )
          break;
      }
      v2 = a2;
      v23 = v82;
    }
    sub_C7D6A0(v81, 32LL * v23, 8);
    v34 = v80;
    if ( v80 )
    {
      v35 = v79;
      v36 = &v79[2 * v80];
      do
      {
        if ( *v35 != -8192 && *v35 != -4096 )
        {
          v37 = v35[1];
          if ( v37 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v37 + 8LL))(v37);
        }
        v35 += 2;
      }
      while ( v36 != v35 );
      v34 = v80;
    }
    sub_C7D6A0((__int64)v79, 16LL * v34, 8);
    v38 = *(__int64 **)(a1 + 8);
    v39 = *v38;
    v40 = v38[1];
    if ( v39 == v40 )
LABEL_76:
      BUG();
    while ( *(_UNKNOWN **)v39 != &unk_4F8662C )
    {
      v39 += 16;
      if ( v40 == v39 )
        goto LABEL_76;
    }
    v41 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v39 + 8) + 104LL))(
            *(_QWORD *)(v39 + 8),
            &unk_4F8662C);
    v42 = sub_CFFAC0(v41, v67);
    v43 = *(__int64 **)(a1 + 8);
    v44 = v42;
    v45 = *v43;
    v46 = v43[1];
    if ( v45 == v46 )
LABEL_77:
      BUG();
    while ( *(_UNKNOWN **)v45 != &unk_4F8144C )
    {
      v45 += 16;
      if ( v46 == v45 )
        goto LABEL_77;
    }
    v47 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v45 + 8) + 104LL))(
            *(_QWORD *)(v45 + 8),
            &unk_4F8144C);
    v48 = *(__int64 **)(a1 + 8);
    v49 = (__int64 *)(v47 + 176);
    v50 = *v48;
    v51 = v48[1];
    if ( v50 == v51 )
LABEL_78:
      BUG();
    while ( *(_UNKNOWN **)v50 != &unk_4F875EC )
    {
      v50 += 16;
      if ( v51 == v50 )
        goto LABEL_78;
    }
    v52 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v50 + 8) + 104LL))(
            *(_QWORD *)(v50 + 8),
            &unk_4F875EC);
    v53 = *(__int64 **)(a1 + 8);
    v54 = v52 + 176;
    v55 = *v53;
    v56 = v53[1];
    if ( v55 == v56 )
LABEL_79:
      BUG();
    while ( *(_UNKNOWN **)v55 != &unk_4F86530 )
    {
      v55 += 16;
      if ( v56 == v55 )
        goto LABEL_79;
    }
    v68 = v54;
    v57 = *(_QWORD **)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v55 + 8) + 104LL))(
                         *(_QWORD *)(v55 + 8),
                         &unk_4F86530)
                     + 176);
    if ( !(unsigned __int8)sub_F6E5B0(v2, (__int64)&unk_4F86530, v58, v59, v60, v61) )
      v70 = sub_27FBD50(v64, v2, v57, v68, v49, v44, v66, v63, v69, v65, v71, 0);
    v62 = v72;
    if ( v72 )
    {
      sub_FDC110(v72);
      j_j___libc_free_0((unsigned __int64)v62);
    }
  }
  return v70;
}
