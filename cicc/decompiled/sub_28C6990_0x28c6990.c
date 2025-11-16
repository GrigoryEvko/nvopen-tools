// Function: sub_28C6990
// Address: 0x28c6990
//
__int64 __fastcall sub_28C6990(_QWORD *a1, __int64 a2)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  __m128i v21; // xmm5
  __m128i v22; // xmm6
  __m128i v23; // xmm7
  __m128i v24; // xmm0
  __m128i v25; // xmm1
  unsigned int v26; // eax
  _QWORD **v27; // rbx
  __int64 v28; // rax
  _QWORD *v29; // r15
  unsigned __int64 v30; // r12
  __int64 v31; // rdi
  __m128i v32; // xmm1
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  __m128i v35; // xmm4
  unsigned int v36; // eax
  _QWORD *v37; // rbx
  _QWORD *v38; // r12
  __int64 v39; // rdi
  __int64 *v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-108h]
  __int64 v46; // [rsp+10h] [rbp-100h]
  __int64 v47; // [rsp+18h] [rbp-F8h]
  __int64 v48; // [rsp+20h] [rbp-F0h]
  _QWORD **v49; // [rsp+28h] [rbp-E8h]
  __m128i v50; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v51; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v52; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v53; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v54; // [rsp+70h] [rbp-A0h] BYREF
  _BYTE v55[8]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD *v56; // [rsp+88h] [rbp-88h]
  unsigned int v57; // [rsp+98h] [rbp-78h]
  __int64 v58; // [rsp+A8h] [rbp-68h]
  unsigned int v59; // [rsp+B8h] [rbp-58h]
  __int64 v60; // [rsp+C8h] [rbp-48h]
  unsigned int v61; // [rsp+D8h] [rbp-38h]

  if ( (unsigned __int8)sub_BB98D0(a1, a2) )
    return 0;
  v4 = (__int64 *)a1[1];
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_54:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4F8662C )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_54;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F8662C);
  v8 = sub_CFFAC0(v7, a2);
  v9 = (__int64 *)a1[1];
  v45 = v8;
  v10 = *v9;
  v11 = v9[1];
  if ( v10 == v11 )
LABEL_55:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4F8144C )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_55;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F8144C);
  v13 = (__int64 *)a1[1];
  v48 = v12 + 176;
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_56:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F881C8 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_56;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F881C8);
  v17 = (__int64 *)a1[1];
  v47 = *(_QWORD *)(v16 + 176);
  v18 = *v17;
  v19 = v17[1];
  if ( v18 == v19 )
LABEL_57:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_4F6D3F0 )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_57;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(*(_QWORD *)(v18 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v55);
  sub_983BD0((__int64)&v50, v20 + 176, a2);
  v46 = v20 + 408;
  if ( *(_BYTE *)(v20 + 488) )
  {
    v32 = _mm_loadu_si128(&v51);
    v33 = _mm_loadu_si128(&v52);
    v34 = _mm_loadu_si128(&v53);
    v35 = _mm_loadu_si128(&v54);
    *(__m128i *)(v20 + 408) = _mm_loadu_si128(&v50);
    *(__m128i *)(v20 + 424) = v32;
    *(__m128i *)(v20 + 440) = v33;
    *(__m128i *)(v20 + 456) = v34;
    *(__m128i *)(v20 + 472) = v35;
  }
  else
  {
    v21 = _mm_loadu_si128(&v50);
    v22 = _mm_loadu_si128(&v51);
    *(_BYTE *)(v20 + 488) = 1;
    v23 = _mm_loadu_si128(&v52);
    v24 = _mm_loadu_si128(&v53);
    v25 = _mm_loadu_si128(&v54);
    *(__m128i *)(v20 + 408) = v21;
    *(__m128i *)(v20 + 424) = v22;
    *(__m128i *)(v20 + 440) = v23;
    *(__m128i *)(v20 + 456) = v24;
    *(__m128i *)(v20 + 472) = v25;
  }
  sub_C7D6A0(v60, 24LL * v61, 8);
  v26 = v59;
  if ( v59 )
  {
    v27 = (_QWORD **)(v58 + 8);
    v49 = (_QWORD **)(v58 + 32LL * v59);
    while ( 1 )
    {
      v28 = (__int64)*(v27 - 1);
      if ( v28 != -8192 && v28 != -4096 )
      {
        v29 = *v27;
        while ( v29 != v27 )
        {
          v30 = (unsigned __int64)v29;
          v29 = (_QWORD *)*v29;
          v31 = *(_QWORD *)(v30 + 24);
          if ( v31 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v31 + 8LL))(v31);
          j_j___libc_free_0(v30);
        }
      }
      if ( v49 == v27 + 3 )
        break;
      v27 += 4;
    }
    v26 = v59;
  }
  sub_C7D6A0(v58, 32LL * v26, 8);
  v36 = v57;
  if ( v57 )
  {
    v37 = v56;
    v38 = &v56[2 * v57];
    do
    {
      if ( *v37 != -4096 && *v37 != -8192 )
      {
        v39 = v37[1];
        if ( v39 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v39 + 8LL))(v39);
      }
      v37 += 2;
    }
    while ( v38 != v37 );
    v36 = v57;
  }
  sub_C7D6A0((__int64)v56, 16LL * v36, 8);
  v40 = (__int64 *)a1[1];
  v41 = *v40;
  v42 = v40[1];
  if ( v41 == v42 )
LABEL_58:
    BUG();
  while ( *(_UNKNOWN **)v41 != &unk_4F89C28 )
  {
    v41 += 16;
    if ( v42 == v41 )
      goto LABEL_58;
  }
  v43 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v41 + 8) + 104LL))(*(_QWORD *)(v41 + 8), &unk_4F89C28);
  v44 = sub_DFED00(v43, a2);
  return sub_28C6930(a1 + 22, a2, v45, v48, v47, v46, v44);
}
