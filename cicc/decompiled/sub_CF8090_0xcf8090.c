// Function: sub_CF8090
// Address: 0xcf8090
//
__int64 __fastcall sub_CF8090(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rbx
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  __m128i v9; // xmm3
  __m128i v10; // xmm4
  unsigned int v11; // eax
  _QWORD **v12; // r12
  _QWORD **i; // rbx
  __int64 v14; // rax
  _QWORD *v15; // r14
  _QWORD *v16; // r15
  __int64 v17; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  __int64 v21; // rdi
  _QWORD *v22; // rax
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // r12
  _QWORD *v31; // rax
  char *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // r12
  _QWORD *v36; // rax
  char *v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 *v40; // r12
  _QWORD *v41; // rax
  char *v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 *v45; // r12
  _QWORD *v46; // rax
  char *v47; // rsi
  __m128i v49; // xmm5
  __m128i v50; // xmm6
  __m128i v51; // xmm7
  __m128i v52; // xmm0
  __m128i v53; // xmm1
  __int64 *v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 *v57; // r12
  _QWORD *v58; // rax
  char *v59; // rsi
  __int64 v60; // [rsp+10h] [rbp-F0h]
  __int64 v61; // [rsp+10h] [rbp-F0h]
  __int64 v62; // [rsp+10h] [rbp-F0h]
  __int64 v63; // [rsp+10h] [rbp-F0h]
  __int64 v64; // [rsp+10h] [rbp-F0h]
  __int64 v65; // [rsp+10h] [rbp-F0h]
  __m128i v67; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v68; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v69; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v70; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v71; // [rsp+60h] [rbp-A0h] BYREF
  _QWORD *v72; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v73; // [rsp+78h] [rbp-88h]
  unsigned int v74; // [rsp+88h] [rbp-78h]
  __int64 v75; // [rsp+98h] [rbp-68h]
  unsigned int v76; // [rsp+A8h] [rbp-58h]
  __int64 v77; // [rsp+B8h] [rbp-48h]
  unsigned int v78; // [rsp+C8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_92:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F6D3F0 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_92;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)&v72);
  sub_983BD0((__int64)&v67, v6 + 176, a2);
  v60 = v6 + 408;
  if ( *(_BYTE *)(v6 + 488) )
  {
    v7 = _mm_loadu_si128(&v68);
    v8 = _mm_loadu_si128(&v69);
    v9 = _mm_loadu_si128(&v70);
    v10 = _mm_loadu_si128(&v71);
    *(__m128i *)(v6 + 408) = _mm_loadu_si128(&v67);
    *(__m128i *)(v6 + 424) = v7;
    *(__m128i *)(v6 + 440) = v8;
    *(__m128i *)(v6 + 456) = v9;
    *(__m128i *)(v6 + 472) = v10;
  }
  else
  {
    v49 = _mm_loadu_si128(&v67);
    v50 = _mm_loadu_si128(&v68);
    *(_BYTE *)(v6 + 488) = 1;
    v51 = _mm_loadu_si128(&v69);
    v52 = _mm_loadu_si128(&v70);
    v53 = _mm_loadu_si128(&v71);
    *(__m128i *)(v6 + 408) = v49;
    *(__m128i *)(v6 + 424) = v50;
    *(__m128i *)(v6 + 440) = v51;
    *(__m128i *)(v6 + 456) = v52;
    *(__m128i *)(v6 + 472) = v53;
  }
  sub_C7D6A0(v77, 24LL * v78, 8);
  v11 = v76;
  if ( v76 )
  {
    v12 = (_QWORD **)(v75 + 32LL * v76);
    for ( i = (_QWORD **)(v75 + 8); ; i += 4 )
    {
      v14 = (__int64)*(i - 1);
      if ( v14 != -4096 && v14 != -8192 )
      {
        v15 = *i;
        while ( v15 != i )
        {
          v16 = v15;
          v15 = (_QWORD *)*v15;
          v17 = v16[3];
          if ( v17 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
          j_j___libc_free_0(v16, 32);
        }
      }
      if ( v12 == i + 3 )
        break;
    }
    v11 = v76;
  }
  sub_C7D6A0(v75, 32LL * v11, 8);
  v18 = v74;
  if ( v74 )
  {
    v19 = v73;
    v20 = &v73[2 * v74];
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
    v18 = v74;
  }
  sub_C7D6A0((__int64)v73, 16LL * v18, 8);
  v22 = (_QWORD *)sub_22077B0(56);
  v23 = v22;
  if ( v22 )
    sub_CF4B40(v22, v60);
  v24 = *(_QWORD **)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v23;
  if ( v24 )
  {
    sub_CF4BF0(v24);
    j_j___libc_free_0(v24, 56);
  }
  v25 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F86538);
  if ( v25
    && (v26 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v25 + 104LL))(v25, &unk_4F86538), (v27 = v26) != 0) )
  {
    if ( *(_BYTE *)(v26 + 208) && *(_QWORD *)(v26 + 192) )
      (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD))(v26 + 200))(v26 + 176, a1, a2, *(_QWORD *)(a1 + 176));
  }
  else
  {
    v27 = 0;
  }
  if ( !byte_4F865E8 )
  {
    v54 = *(__int64 **)(a1 + 8);
    v55 = *v54;
    v56 = v54[1];
    if ( v55 == v56 )
LABEL_93:
      BUG();
    while ( *(_UNKNOWN **)v55 != &unk_4F8670C )
    {
      v55 += 16;
      if ( v56 == v55 )
        goto LABEL_93;
    }
    v57 = *(__int64 **)(a1 + 176);
    v65 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v55 + 8) + 104LL))(
                        *(_QWORD *)(v55 + 8),
                        &unk_4F8670C)
                    + 176);
    v58 = (_QWORD *)sub_22077B0(16);
    if ( v58 )
    {
      v58[1] = v65;
      *v58 = &unk_49DD930;
    }
    v72 = v58;
    v59 = (char *)v57[2];
    if ( v59 == (char *)v57[3] )
    {
      sub_CF7870(v57 + 1, v59, &v72);
    }
    else
    {
      if ( v59 )
      {
        *(_QWORD *)v59 = v58;
        v59 = (char *)v57[2];
      }
      v57[2] = (__int64)(v59 + 8);
    }
  }
  v28 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F89B44);
  if ( v28 )
  {
    v29 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v28 + 104LL))(v28, &unk_4F89B44);
    if ( v29 )
    {
      v30 = *(__int64 **)(a1 + 176);
      v61 = *(_QWORD *)(v29 + 176);
      v31 = (_QWORD *)sub_22077B0(16);
      if ( v31 )
      {
        v31[1] = v61;
        *v31 = &unk_49DD988;
      }
      v72 = v31;
      v32 = (char *)v30[2];
      if ( v32 == (char *)v30[3] )
      {
        sub_CF7A10(v30 + 1, v32, &v72);
      }
      else
      {
        if ( v32 )
        {
          *(_QWORD *)v32 = v31;
          v32 = (char *)v30[2];
        }
        v30[2] = (__int64)(v32 + 8);
      }
    }
  }
  v33 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F89FAC);
  if ( v33 )
  {
    v34 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v33 + 104LL))(v33, &unk_4F89FAC);
    if ( v34 )
    {
      v35 = *(__int64 **)(a1 + 176);
      v62 = *(_QWORD *)(v34 + 176);
      v36 = (_QWORD *)sub_22077B0(16);
      if ( v36 )
      {
        v36[1] = v62;
        *v36 = &unk_49DD9E0;
      }
      v72 = v36;
      v37 = (char *)v35[2];
      if ( v37 == (char *)v35[3] )
      {
        sub_CF7BB0(v35 + 1, v37, &v72);
      }
      else
      {
        if ( v37 )
        {
          *(_QWORD *)v37 = v36;
          v37 = (char *)v35[2];
        }
        v35[2] = (__int64)(v37 + 8);
      }
    }
  }
  v38 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F86B74);
  if ( v38 )
  {
    v39 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v38 + 104LL))(v38, &unk_4F86B74);
    if ( v39 )
    {
      v40 = *(__int64 **)(a1 + 176);
      v63 = *(_QWORD *)(v39 + 176);
      v41 = (_QWORD *)sub_22077B0(16);
      if ( v41 )
      {
        v41[1] = v63;
        *v41 = &unk_49DDA38;
      }
      v72 = v41;
      v42 = (char *)v40[2];
      if ( v42 == (char *)v40[3] )
      {
        sub_CF7D50(v40 + 1, v42, &v72);
      }
      else
      {
        if ( v42 )
        {
          *(_QWORD *)v42 = v41;
          v42 = (char *)v40[2];
        }
        v40[2] = (__int64)(v42 + 8);
      }
    }
  }
  v43 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F89B30);
  if ( v43 )
  {
    v44 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v43 + 104LL))(v43, &unk_4F89B30);
    if ( v44 )
    {
      v45 = *(__int64 **)(a1 + 176);
      v64 = *(_QWORD *)(v44 + 176);
      v46 = (_QWORD *)sub_22077B0(16);
      if ( v46 )
      {
        v46[1] = v64;
        *v46 = &unk_49DDA90;
      }
      v72 = v46;
      v47 = (char *)v45[2];
      if ( v47 == (char *)v45[3] )
      {
        sub_CF7EF0(v45 + 1, v47, &v72);
      }
      else
      {
        if ( v47 )
        {
          *(_QWORD *)v47 = v46;
          v47 = (char *)v45[2];
        }
        v45[2] = (__int64)(v47 + 8);
      }
    }
  }
  if ( v27 && !*(_BYTE *)(v27 + 208) && *(_QWORD *)(v27 + 192) )
    (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD))(v27 + 200))(v27 + 176, a1, a2, *(_QWORD *)(a1 + 176));
  return 0;
}
