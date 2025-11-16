// Function: sub_27A0740
// Address: 0x27a0740
//
__int64 __fastcall sub_27A0740(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // r13
  char v4; // r8
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 v7; // rbx
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
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rbx
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v26; // xmm4
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned int v32; // eax
  _QWORD **v33; // r15
  _QWORD **i; // rbx
  __int64 v35; // rax
  _QWORD *v36; // r12
  unsigned __int64 v37; // r13
  __int64 v38; // rdi
  __int64 *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned int v45; // eax
  _QWORD *v46; // rbx
  _QWORD *v47; // r12
  __int64 v48; // rdi
  __int64 *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 *v53; // rdx
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  __m128i v59; // xmm5
  __m128i v60; // xmm6
  __m128i v61; // xmm7
  __m128i v62; // xmm0
  __m128i v63; // xmm1
  __int64 v64; // [rsp+0h] [rbp-120h]
  __int64 v65; // [rsp+8h] [rbp-118h]
  __int64 v66; // [rsp+10h] [rbp-110h]
  __int64 v67; // [rsp+18h] [rbp-108h]
  __int64 v68; // [rsp+20h] [rbp-100h]
  _QWORD *v69; // [rsp+28h] [rbp-F8h]
  __int64 v70; // [rsp+30h] [rbp-F0h]
  __int64 v71; // [rsp+38h] [rbp-E8h]
  __m128i v72; // [rsp+40h] [rbp-E0h] BYREF
  __m128i v73; // [rsp+50h] [rbp-D0h] BYREF
  __m128i v74; // [rsp+60h] [rbp-C0h] BYREF
  __m128i v75; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v76; // [rsp+80h] [rbp-A0h] BYREF
  char v77[8]; // [rsp+90h] [rbp-90h] BYREF
  _QWORD *v78; // [rsp+98h] [rbp-88h]
  unsigned int v79; // [rsp+A8h] [rbp-78h]
  __int64 v80; // [rsp+B8h] [rbp-68h]
  unsigned int v81; // [rsp+C8h] [rbp-58h]
  __int64 v82; // [rsp+D8h] [rbp-48h]
  unsigned int v83; // [rsp+E8h] [rbp-38h]

  v3 = a1;
  v4 = sub_BB98D0(a1, a2);
  result = 0;
  if ( v4 )
    return result;
  v6 = sub_B82360(a1[1], (__int64)&unk_4F8F808);
  v71 = (__int64)(v3 + 22);
  if ( v6 )
  {
    v7 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v6 + 104LL))(v6, &unk_4F8F808);
    if ( !(unsigned __int8)sub_278A9C0(v71) )
      goto LABEL_42;
    if ( v7 )
    {
LABEL_5:
      v70 = *(_QWORD *)(v7 + 176);
      goto LABEL_6;
    }
  }
  else if ( !(unsigned __int8)sub_278A9C0((__int64)(v3 + 22)) )
  {
    goto LABEL_43;
  }
  v39 = (__int64 *)v3[1];
  v40 = *v39;
  v41 = v39[1];
  if ( v40 == v41 )
    goto LABEL_68;
  while ( *(_UNKNOWN **)v40 != &unk_4F8F808 )
  {
    v40 += 16;
    if ( v41 == v40 )
      goto LABEL_68;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v40 + 8) + 104LL))(*(_QWORD *)(v40 + 8), &unk_4F8F808);
LABEL_42:
  if ( v7 )
    goto LABEL_5;
LABEL_43:
  v70 = 0;
LABEL_6:
  v8 = (__int64 *)v3[1];
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
    goto LABEL_68;
  while ( *(_UNKNOWN **)v9 != &unk_4F8FAE4 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_68;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F8FAE4);
  v12 = (__int64 *)v3[1];
  v65 = *(_QWORD *)(v11 + 176);
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
    goto LABEL_68;
  while ( *(_UNKNOWN **)v13 != &unk_4F875EC )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_68;
  }
  v68 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F875EC)
      + 176;
  if ( (unsigned __int8)sub_278A9A0(v71) )
  {
    v42 = (__int64 *)v3[1];
    v43 = *v42;
    v44 = v42[1];
    if ( v43 == v44 )
      goto LABEL_68;
    while ( *(_UNKNOWN **)v43 != &unk_4F8EE5C )
    {
      v43 += 16;
      if ( v44 == v43 )
        goto LABEL_68;
    }
    v66 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v43 + 8) + 104LL))(
            *(_QWORD *)(v43 + 8),
            &unk_4F8EE5C)
        + 176;
  }
  else
  {
    v66 = 0;
  }
  v15 = (__int64 *)v3[1];
  v16 = *v15;
  v17 = v15[1];
  if ( v16 == v17 )
    goto LABEL_68;
  while ( *(_UNKNOWN **)v16 != &unk_4F86530 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_68;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F86530);
  v19 = (__int64 *)v3[1];
  v64 = *(_QWORD *)(v18 + 176);
  v20 = *v19;
  v21 = v19[1];
  if ( v20 == v21 )
    goto LABEL_68;
  while ( *(_UNKNOWN **)v20 != &unk_4F6D3F0 )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_68;
  }
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v77);
  sub_983BD0((__int64)&v72, v22 + 176, a2);
  v67 = v22 + 408;
  if ( *(_BYTE *)(v22 + 488) )
  {
    v23 = _mm_loadu_si128(&v73);
    v24 = _mm_loadu_si128(&v74);
    v25 = _mm_loadu_si128(&v75);
    v26 = _mm_loadu_si128(&v76);
    *(__m128i *)(v22 + 408) = _mm_loadu_si128(&v72);
    *(__m128i *)(v22 + 424) = v23;
    *(__m128i *)(v22 + 440) = v24;
    *(__m128i *)(v22 + 456) = v25;
    *(__m128i *)(v22 + 472) = v26;
  }
  else
  {
    v59 = _mm_loadu_si128(&v72);
    v60 = _mm_loadu_si128(&v73);
    *(_BYTE *)(v22 + 488) = 1;
    v61 = _mm_loadu_si128(&v74);
    v62 = _mm_loadu_si128(&v75);
    v63 = _mm_loadu_si128(&v76);
    *(__m128i *)(v22 + 408) = v59;
    *(__m128i *)(v22 + 424) = v60;
    *(__m128i *)(v22 + 440) = v61;
    *(__m128i *)(v22 + 456) = v62;
    *(__m128i *)(v22 + 472) = v63;
  }
  v27 = 24LL * v83;
  sub_C7D6A0(v82, v27, 8);
  v32 = v81;
  if ( v81 )
  {
    v69 = v3;
    v33 = (_QWORD **)(v80 + 32LL * v81);
    for ( i = (_QWORD **)(v80 + 8); ; i += 4 )
    {
      v35 = (__int64)*(i - 1);
      if ( v35 != -8192 && v35 != -4096 )
      {
        v36 = *i;
        while ( v36 != i )
        {
          v37 = (unsigned __int64)v36;
          v36 = (_QWORD *)*v36;
          v38 = *(_QWORD *)(v37 + 24);
          if ( v38 )
            (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v38 + 8LL))(
              v38,
              v27,
              v28,
              v29,
              v30,
              v31,
              v64,
              v65,
              v66,
              v67,
              v68);
          v27 = 32;
          j_j___libc_free_0(v37);
        }
      }
      if ( v33 == i + 3 )
        break;
    }
    v3 = v69;
    v32 = v81;
  }
  sub_C7D6A0(v80, 32LL * v32, 8);
  v45 = v79;
  if ( v79 )
  {
    v46 = v78;
    v47 = &v78[2 * v79];
    do
    {
      if ( *v46 != -8192 && *v46 != -4096 )
      {
        v48 = v46[1];
        if ( v48 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v48 + 8LL))(v48);
      }
      v46 += 2;
    }
    while ( v47 != v46 );
    v45 = v79;
  }
  sub_C7D6A0((__int64)v78, 16LL * v45, 8);
  v49 = (__int64 *)v3[1];
  v50 = *v49;
  v51 = v49[1];
  if ( v50 == v51 )
    goto LABEL_68;
  while ( *(_UNKNOWN **)v50 != &unk_4F8144C )
  {
    v50 += 16;
    if ( v51 == v50 )
      goto LABEL_68;
  }
  v52 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v50 + 8) + 104LL))(*(_QWORD *)(v50 + 8), &unk_4F8144C);
  v53 = (__int64 *)v3[1];
  v54 = v52 + 176;
  v55 = *v53;
  v56 = v53[1];
  if ( v55 == v56 )
LABEL_68:
    BUG();
  while ( *(_UNKNOWN **)v55 != &unk_4F8662C )
  {
    v55 += 16;
    if ( v56 == v55 )
      goto LABEL_68;
  }
  v57 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v55 + 8) + 104LL))(*(_QWORD *)(v55 + 8), &unk_4F8662C);
  v58 = sub_CFFAC0(v57, a2);
  return sub_279FBF0(v71, a2, v58, v54, v67, v64, v66, v68, v65, v70);
}
