// Function: sub_2362620
// Address: 0x2362620
//
__int64 *__fastcall sub_2362620(__int64 a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 *v5; // rbx
  int v6; // edx
  int v7; // ecx
  __int64 *v8; // rbx
  int v9; // r8d
  int v10; // r9d
  __int64 *v11; // rbx
  __int64 *v12; // rbx
  __int64 *v13; // rbx
  __int64 *v14; // rbx
  __int64 *v15; // rbx
  __int64 *v16; // rbx
  __int64 *v17; // rbx
  __int64 *v18; // rbx
  __int64 *v19; // rbx
  __int64 *v20; // rbx
  __int64 *v21; // rbx
  __int64 *v22; // rbx
  __int64 *v23; // rbx
  __int64 *v24; // rbx
  __m128i *v25; // rsi
  __int64 v26; // rdi
  __int64 *result; // rax
  void *v28; // rdx
  __int64 *v29; // rbx
  __int64 v30; // rbx
  __int64 i; // r13
  __int64 *v32; // rbx
  _QWORD *v33; // rax
  __int64 v34; // rdi
  _QWORD *v35; // rax
  __int64 v36; // rdi
  _QWORD *v37; // rax
  __int64 v38; // rdi
  _QWORD *v39; // rax
  __int64 v40; // rdi
  _QWORD *v41; // rax
  __int64 v42; // rdi
  __int64 v43; // r15
  _QWORD *v44; // rax
  __int64 v45; // rdi
  _QWORD *v46; // rax
  __int64 v47; // rdi
  _QWORD *v48; // rax
  __int64 v49; // rdi
  _QWORD *v50; // rax
  __int64 v51; // rdi
  _QWORD *v52; // rax
  __int64 v53; // rdi
  _QWORD *v54; // rax
  __int64 v55; // rdi
  _QWORD *v56; // rax
  __int64 v57; // rdi
  _QWORD *v58; // rax
  __int64 v59; // rdi
  _QWORD *v60; // rax
  __int64 v61; // rdi
  _QWORD *v62; // rax
  __int64 v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rdx
  __m128i v66; // xmm0
  __int64 v67; // rdi
  _QWORD *v68; // rax
  __int64 v69; // rdi
  __m128i v70; // [rsp+20h] [rbp-50h] BYREF
  __int64 v71; // [rsp+30h] [rbp-40h]

  v70.m128i_i64[0] = (__int64)&unk_4F86A90;
  v4 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v4 )
  {
    v32 = v4;
    v33 = (_QWORD *)sub_22077B0(0x10u);
    if ( v33 )
      *v33 = &unk_4A0B6E8;
    v34 = *v32;
    *v32 = (__int64)v33;
    if ( v34 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 8LL))(v34);
  }
  v70.m128i_i64[0] = (__int64)&unk_501DA18;
  v5 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v5 )
  {
    v68 = (_QWORD *)sub_22077B0(0x10u);
    if ( v68 )
      *v68 = &unk_4A0B718;
    v69 = *v5;
    *v5 = (__int64)v68;
    if ( v69 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v69 + 8LL))(v69);
  }
  v70.m128i_i64[0] = (__int64)&unk_502E1A8;
  v8 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v8 )
  {
    sub_30A7B40((unsigned int)&v70, (unsigned int)&v70, v6, v7, v9, v10);
    v64 = sub_22077B0(0x20u);
    if ( v64 )
    {
      v65 = v71;
      v66 = _mm_loadu_si128(&v70);
      *(_QWORD *)v64 = &unk_4A0B748;
      *(_QWORD *)(v64 + 24) = v65;
      *(__m128i *)(v64 + 8) = v66;
    }
    v67 = *v8;
    *v8 = v64;
    if ( v67 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v67 + 8LL))(v67);
  }
  v70.m128i_i64[0] = (__int64)&unk_502ED88;
  v11 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v11 )
  {
    v62 = (_QWORD *)sub_22077B0(0x10u);
    if ( v62 )
      *v62 = &unk_4A0B778;
    v63 = *v11;
    *v11 = (__int64)v62;
    if ( v63 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v63 + 8LL))(v63);
  }
  v70.m128i_i64[0] = (__int64)&unk_4FDB698;
  v12 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v12 )
  {
    v60 = (_QWORD *)sub_22077B0(0x10u);
    if ( v60 )
      *v60 = &unk_4A0B7A8;
    v61 = *v12;
    *v12 = (__int64)v60;
    if ( v61 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v61 + 8LL))(v61);
  }
  v70.m128i_i64[0] = (__int64)&unk_4FDB6A0;
  v13 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v13 )
  {
    v58 = (_QWORD *)sub_22077B0(0x10u);
    if ( v58 )
      *v58 = &unk_4A0B7D8;
    v59 = *v13;
    *v13 = (__int64)v58;
    if ( v59 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v59 + 8LL))(v59);
  }
  v70.m128i_i64[0] = (__int64)&unk_502F110;
  v14 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v14 )
  {
    v56 = (_QWORD *)sub_22077B0(0x10u);
    if ( v56 )
      *v56 = &unk_4A0B808;
    v57 = *v14;
    *v14 = (__int64)v56;
    if ( v57 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v57 + 8LL))(v57);
  }
  v70.m128i_i64[0] = (__int64)&unk_4FDB950;
  v15 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v15 )
  {
    v54 = (_QWORD *)sub_22077B0(0x10u);
    if ( v54 )
      *v54 = &unk_4A0B838;
    v55 = *v15;
    *v15 = (__int64)v54;
    if ( v55 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v55 + 8LL))(v55);
  }
  v70.m128i_i64[0] = (__int64)&unk_4F8ED68;
  v16 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v16 )
  {
    v52 = (_QWORD *)sub_22077B0(0x10u);
    if ( v52 )
      *v52 = &unk_4A0B868;
    v53 = *v16;
    *v16 = (__int64)v52;
    if ( v53 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v53 + 8LL))(v53);
  }
  v70.m128i_i64[0] = (__int64)&unk_4F86C48;
  v17 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v17 )
  {
    v50 = (_QWORD *)sub_22077B0(0x10u);
    if ( v50 )
      *v50 = &unk_4A0B898;
    v51 = *v17;
    *v17 = (__int64)v50;
    if ( v51 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v51 + 8LL))(v51);
  }
  v70.m128i_i64[0] = (__int64)&unk_4F87818;
  v18 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v18 )
  {
    v48 = (_QWORD *)sub_22077B0(0x10u);
    if ( v48 )
      *v48 = &unk_4A0B8C8;
    v49 = *v18;
    *v18 = (__int64)v48;
    if ( v49 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v49 + 8LL))(v49);
  }
  v70.m128i_i64[0] = (__int64)&unk_4FDC288;
  v19 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v19 )
  {
    v46 = (_QWORD *)sub_22077B0(0x10u);
    if ( v46 )
      *v46 = &unk_4A0B8F8;
    v47 = *v19;
    *v19 = (__int64)v46;
    if ( v47 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v47 + 8LL))(v47);
  }
  v70.m128i_i64[0] = (__int64)&qword_4F8A320;
  v20 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v20 )
  {
    v43 = *(_QWORD *)(a1 + 200);
    v44 = (_QWORD *)sub_22077B0(0x10u);
    if ( v44 )
    {
      v44[1] = v43;
      *v44 = &unk_4A0B928;
    }
    v45 = *v20;
    *v20 = (__int64)v44;
    if ( v45 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v45 + 8LL))(v45);
  }
  v70.m128i_i64[0] = (__int64)&unk_4F87C68;
  v21 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v21 )
  {
    v41 = (_QWORD *)sub_22077B0(0x10u);
    if ( v41 )
      *v41 = &unk_4A0B958;
    v42 = *v21;
    *v21 = (__int64)v41;
    if ( v42 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v42 + 8LL))(v42);
  }
  v70.m128i_i64[0] = (__int64)&unk_5024E68;
  v22 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v22 )
  {
    v39 = (_QWORD *)sub_22077B0(0x10u);
    if ( v39 )
      *v39 = &unk_4A0B988;
    v40 = *v22;
    *v22 = (__int64)v39;
    if ( v40 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v40 + 8LL))(v40);
  }
  v70.m128i_i64[0] = (__int64)&unk_4F87F18;
  v23 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v23 )
  {
    v37 = (_QWORD *)sub_22077B0(0x10u);
    if ( v37 )
      *v37 = &unk_4A0B9B8;
    v38 = *v23;
    *v23 = (__int64)v37;
    if ( v38 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v38 + 8LL))(v38);
  }
  v70.m128i_i64[0] = (__int64)&unk_4F836C8;
  v24 = sub_23624E0(a2, v70.m128i_i64);
  if ( !*v24 )
  {
    v35 = (_QWORD *)sub_22077B0(0x10u);
    if ( v35 )
      *v35 = &unk_4A0B9E8;
    v36 = *v24;
    *v24 = (__int64)v35;
    if ( v36 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
  }
  v25 = &v70;
  v26 = a2;
  v70.m128i_i64[0] = (__int64)&unk_4F86B78;
  result = sub_23624E0(a2, v70.m128i_i64);
  v29 = result;
  if ( !*result )
  {
    result = (__int64 *)sub_22077B0(0x10u);
    if ( result )
    {
      v28 = &unk_4A0BA18;
      *result = (__int64)&unk_4A0BA18;
    }
    v26 = *v29;
    *v29 = (__int64)result;
    if ( v26 )
      result = (__int64 *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
  }
  v30 = *(_QWORD *)(a1 + 1248);
  for ( i = v30 + 32LL * *(unsigned int *)(a1 + 1256); v30 != i; v30 += 32 )
  {
    if ( !*(_QWORD *)(v30 + 16) )
      sub_4263D6(v26, v25, v28);
    v26 = v30;
    v25 = (__m128i *)a2;
    result = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(v30 + 24))(v30, a2);
  }
  return result;
}
