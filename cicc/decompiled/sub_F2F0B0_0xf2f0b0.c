// Function: sub_F2F0B0
// Address: 0xf2f0b0
//
__int64 __fastcall sub_F2F0B0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __m128i v17; // xmm5
  __m128i v18; // xmm6
  __m128i v19; // xmm7
  __m128i v20; // xmm0
  __m128i v21; // xmm1
  unsigned int v22; // eax
  _QWORD **v23; // r15
  _QWORD **i; // rbx
  __int64 v25; // rax
  _QWORD *v26; // r13
  _QWORD *v27; // r12
  __int64 v28; // rdi
  __m128i v29; // xmm1
  __m128i v30; // xmm2
  __m128i v31; // xmm3
  __m128i v32; // xmm4
  unsigned int v33; // eax
  _QWORD *v34; // rbx
  _QWORD *v35; // r13
  __int64 v36; // rdi
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 *v42; // rdx
  __int64 v43; // r15
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 *v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 *v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // r13
  __int64 *v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // r11
  __int64 v65; // r10
  __int64 v66; // [rsp+8h] [rbp-128h]
  __int64 v67; // [rsp+10h] [rbp-120h]
  __int64 v68; // [rsp+18h] [rbp-118h]
  __int64 v69; // [rsp+18h] [rbp-118h]
  __int64 v70; // [rsp+20h] [rbp-110h]
  __int64 v71; // [rsp+30h] [rbp-100h]
  __int64 v72; // [rsp+38h] [rbp-F8h]
  __int64 v74; // [rsp+40h] [rbp-F0h]
  __int64 v75; // [rsp+48h] [rbp-E8h]
  __m128i v76; // [rsp+50h] [rbp-E0h] BYREF
  __m128i v77; // [rsp+60h] [rbp-D0h] BYREF
  __m128i v78; // [rsp+70h] [rbp-C0h] BYREF
  __m128i v79; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v80; // [rsp+90h] [rbp-A0h] BYREF
  char v81[4]; // [rsp+A0h] [rbp-90h] BYREF
  int v82; // [rsp+A4h] [rbp-8Ch]
  _QWORD *v83; // [rsp+A8h] [rbp-88h]
  unsigned int v84; // [rsp+B8h] [rbp-78h]
  __int64 v85; // [rsp+C8h] [rbp-68h]
  unsigned int v86; // [rsp+D8h] [rbp-58h]
  __int64 v87; // [rsp+E8h] [rbp-48h]
  unsigned int v88; // [rsp+F8h] [rbp-38h]

  v2 = a1;
  if ( (unsigned __int8)sub_BB98D0(a1, a2) )
    return 0;
  v4 = (__int64 *)a1[1];
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_85:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4F86530 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_85;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F86530);
  v8 = (__int64 *)a1[1];
  v70 = *(_QWORD *)(v7 + 176);
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
LABEL_82:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F8662C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_82;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F8662C);
  v12 = sub_CFFAC0(v11, a2);
  v13 = (__int64 *)a1[1];
  v75 = v12;
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_83:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F6D3F0 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_83;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v81);
  sub_983BD0((__int64)&v76, v16 + 176, a2);
  v71 = v16 + 408;
  if ( *(_BYTE *)(v16 + 488) )
  {
    v29 = _mm_loadu_si128(&v77);
    v30 = _mm_loadu_si128(&v78);
    v31 = _mm_loadu_si128(&v79);
    v32 = _mm_loadu_si128(&v80);
    *(__m128i *)(v16 + 408) = _mm_loadu_si128(&v76);
    *(__m128i *)(v16 + 424) = v29;
    *(__m128i *)(v16 + 440) = v30;
    *(__m128i *)(v16 + 456) = v31;
    *(__m128i *)(v16 + 472) = v32;
  }
  else
  {
    v17 = _mm_loadu_si128(&v76);
    v18 = _mm_loadu_si128(&v77);
    *(_BYTE *)(v16 + 488) = 1;
    v19 = _mm_loadu_si128(&v78);
    v20 = _mm_loadu_si128(&v79);
    v21 = _mm_loadu_si128(&v80);
    *(__m128i *)(v16 + 408) = v17;
    *(__m128i *)(v16 + 424) = v18;
    *(__m128i *)(v16 + 440) = v19;
    *(__m128i *)(v16 + 456) = v20;
    *(__m128i *)(v16 + 472) = v21;
  }
  sub_C7D6A0(v87, 24LL * v88, 8);
  v22 = v86;
  if ( v86 )
  {
    v23 = (_QWORD **)(v85 + 32LL * v86);
    for ( i = (_QWORD **)(v85 + 8); ; i += 4 )
    {
      v25 = (__int64)*(i - 1);
      if ( v25 != -8192 && v25 != -4096 )
      {
        v26 = *i;
        while ( v26 != i )
        {
          v27 = v26;
          v26 = (_QWORD *)*v26;
          v28 = v27[3];
          if ( v28 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v28 + 8LL))(v28);
          j_j___libc_free_0(v27, 32);
        }
      }
      if ( v23 == i + 3 )
        break;
    }
    v2 = a1;
    v22 = v86;
  }
  sub_C7D6A0(v85, 32LL * v22, 8);
  v33 = v84;
  if ( v84 )
  {
    v34 = v83;
    v35 = &v83[2 * v84];
    do
    {
      if ( *v34 != -8192 && *v34 != -4096 )
      {
        v36 = v34[1];
        if ( v36 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
      }
      v34 += 2;
    }
    while ( v35 != v34 );
    v33 = v84;
  }
  sub_C7D6A0((__int64)v83, 16LL * v33, 8);
  v37 = (__int64 *)v2[1];
  v38 = *v37;
  v39 = v37[1];
  if ( v38 == v39 )
LABEL_84:
    BUG();
  while ( *(_UNKNOWN **)v38 != &unk_4F89C28 )
  {
    v38 += 16;
    if ( v39 == v38 )
      goto LABEL_84;
  }
  v40 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v38 + 8) + 104LL))(*(_QWORD *)(v38 + 8), &unk_4F89C28);
  v41 = sub_DFED00(v40, a2);
  v42 = (__int64 *)v2[1];
  v43 = v41;
  v44 = *v42;
  v45 = v42[1];
  if ( v44 == v45 )
LABEL_87:
    BUG();
  while ( *(_UNKNOWN **)v44 != &unk_4F8144C )
  {
    v44 += 16;
    if ( v45 == v44 )
      goto LABEL_87;
  }
  v46 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v44 + 8) + 104LL))(*(_QWORD *)(v44 + 8), &unk_4F8144C);
  v47 = (__int64 *)v2[1];
  v74 = v46 + 176;
  v48 = *v47;
  v49 = v47[1];
  if ( v48 == v49 )
LABEL_88:
    BUG();
  while ( *(_UNKNOWN **)v48 != &unk_4F8FAE4 )
  {
    v48 += 16;
    if ( v49 == v48 )
      goto LABEL_88;
  }
  v50 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v48 + 8) + 104LL))(*(_QWORD *)(v48 + 8), &unk_4F8FAE4);
  v51 = (__int64 *)v2[1];
  v72 = *(_QWORD *)(v50 + 176);
  v52 = *v51;
  v53 = v51[1];
  if ( v52 == v53 )
LABEL_86:
    BUG();
  while ( *(_UNKNOWN **)v52 != &unk_4F87C64 )
  {
    v52 += 16;
    if ( v53 == v52 )
      goto LABEL_86;
  }
  v54 = 0;
  v55 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v52 + 8) + 104LL))(
                      *(_QWORD *)(v52 + 8),
                      &unk_4F87C64)
                  + 176);
  if ( v55 )
  {
    v54 = *(_QWORD *)(v55 + 8);
    if ( v54 )
    {
      v56 = (__int64 *)v2[1];
      v57 = *v56;
      v58 = v56[1];
      if ( v57 == v58 )
LABEL_81:
        BUG();
      while ( *(_UNKNOWN **)v57 != &unk_4F8EE48 )
      {
        v57 += 16;
        if ( v58 == v57 )
          goto LABEL_81;
      }
      v59 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v57 + 8) + 104LL))(
              *(_QWORD *)(v57 + 8),
              &unk_4F8EE48);
      v54 = v59 + 176;
      if ( !*(_BYTE *)(v59 + 184) )
      {
        v63 = *(_QWORD *)(v59 + 200);
        v64 = *(_QWORD *)(v59 + 208);
        v65 = *(_QWORD *)(v63 + 176);
        if ( !*(_BYTE *)(v65 + 280) )
        {
          v66 = v59;
          v67 = *(_QWORD *)(v59 + 208);
          v69 = *(_QWORD *)(v63 + 176);
          sub_FF9360(v69, *(_QWORD *)(v65 + 288), *(_QWORD *)(v65 + 296), *(_QWORD *)(v65 + 304), 0, 0);
          v65 = v69;
          v59 = v66;
          v64 = v67;
          *(_BYTE *)(v69 + 280) = 1;
        }
        v68 = v59;
        sub_FE7D70(v54, *(_QWORD *)(v59 + 192), v65, v64);
        *(_BYTE *)(v68 + 184) = 1;
      }
    }
  }
  v60 = sub_B82360(v2[1], (__int64)&unk_4F8E808);
  if ( v60 && (v61 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v60 + 104LL))(v60, &unk_4F8E808)) != 0 )
    v62 = v61 + 176;
  else
    v62 = 0;
  v81[0] = 0;
  LOBYTE(v83) = 0;
  v82 = 1;
  return sub_F2DD30(a2, (__int64)(v2 + 22), v70, v75, v71, v43, v74, v72, v54, v62, v55, v81);
}
