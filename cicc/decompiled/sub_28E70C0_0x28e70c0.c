// Function: sub_28E70C0
// Address: 0x28e70c0
//
__int64 __fastcall sub_28E70C0(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  unsigned int v23; // eax
  _QWORD **v24; // r14
  _QWORD **i; // r13
  __int64 v26; // rax
  _QWORD *v27; // r12
  unsigned __int64 v28; // r15
  __int64 v29; // rdi
  unsigned int v30; // eax
  _QWORD *v31; // r12
  _QWORD *v32; // r13
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 *v39; // r13
  __int64 v40; // r14
  __int64 *v41; // r12
  __int64 *j; // r15
  __int64 v43; // rsi
  __m128i v45; // xmm5
  __m128i v46; // xmm6
  __m128i v47; // xmm7
  __m128i v48; // xmm0
  __m128i v49; // xmm1
  __int64 v50; // [rsp+8h] [rbp-E8h]
  __int64 v51; // [rsp+8h] [rbp-E8h]
  __m128i v52; // [rsp+10h] [rbp-E0h] BYREF
  __m128i v53; // [rsp+20h] [rbp-D0h] BYREF
  __m128i v54; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v55; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v56; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE v57[8]; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v58; // [rsp+68h] [rbp-88h]
  unsigned int v59; // [rsp+78h] [rbp-78h]
  __int64 v60; // [rsp+88h] [rbp-68h]
  unsigned int v61; // [rsp+98h] [rbp-58h]
  __int64 v62; // [rsp+A8h] [rbp-48h]
  unsigned int v63; // [rsp+B8h] [rbp-38h]

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_54:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F881C8 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_54;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F881C8);
  v7 = (__int64 *)a1[1];
  a1[26] = *(_QWORD *)(v6 + 176);
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_51:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F8144C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_51;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F8144C);
  v11 = (__int64 *)a1[1];
  a1[27] = v10 + 176;
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_52:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F875EC )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_52;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F875EC);
  v15 = (__int64 *)a1[1];
  a1[28] = v14 + 176;
  v16 = *v15;
  v17 = v15[1];
  if ( v16 == v17 )
LABEL_53:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F6D3F0 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_53;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v57);
  sub_983BD0((__int64)&v52, v18 + 176, a2);
  v50 = v18 + 408;
  if ( *(_BYTE *)(v18 + 488) )
  {
    v19 = _mm_loadu_si128(&v53);
    v20 = _mm_loadu_si128(&v54);
    v21 = _mm_loadu_si128(&v55);
    v22 = _mm_loadu_si128(&v56);
    *(__m128i *)(v18 + 408) = _mm_loadu_si128(&v52);
    *(__m128i *)(v18 + 424) = v19;
    *(__m128i *)(v18 + 440) = v20;
    *(__m128i *)(v18 + 456) = v21;
    *(__m128i *)(v18 + 472) = v22;
  }
  else
  {
    v45 = _mm_loadu_si128(&v52);
    v46 = _mm_loadu_si128(&v53);
    *(_BYTE *)(v18 + 488) = 1;
    v47 = _mm_loadu_si128(&v54);
    v48 = _mm_loadu_si128(&v55);
    v49 = _mm_loadu_si128(&v56);
    *(__m128i *)(v18 + 408) = v45;
    *(__m128i *)(v18 + 424) = v46;
    *(__m128i *)(v18 + 440) = v47;
    *(__m128i *)(v18 + 456) = v48;
    *(__m128i *)(v18 + 472) = v49;
  }
  sub_C7D6A0(v62, 24LL * v63, 8);
  v23 = v61;
  if ( v61 )
  {
    v24 = (_QWORD **)(v60 + 32LL * v61);
    for ( i = (_QWORD **)(v60 + 8); ; i += 4 )
    {
      v26 = (__int64)*(i - 1);
      if ( v26 != -8192 && v26 != -4096 )
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
    v23 = v61;
  }
  sub_C7D6A0(v60, 32LL * v23, 8);
  v30 = v59;
  if ( v59 )
  {
    v31 = v58;
    v32 = &v58[2 * v59];
    do
    {
      if ( *v31 != -8192 && *v31 != -4096 )
      {
        v33 = v31[1];
        if ( v33 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v33 + 8LL))(v33);
      }
      v31 += 2;
    }
    while ( v32 != v31 );
    v30 = v59;
  }
  sub_C7D6A0((__int64)v58, 16LL * v30, 8);
  a1[29] = v50;
  v37 = a1[28];
  v38 = *(_QWORD *)(v37 + 40);
  v39 = *(__int64 **)(v37 + 32);
  v51 = v38;
  while ( (__int64 *)v51 != v39 )
  {
    v40 = *v39;
    v41 = *(__int64 **)(*v39 + 16);
    for ( j = *(__int64 **)(*v39 + 8); v41 != j; ++j )
    {
      v43 = *j;
      sub_28E6DF0((__int64)a1, v43, v34, v38, v35, v36);
    }
    ++v39;
    sub_28E6650((__int64)a1, v40, v34, v38, v35, v36);
  }
  return 0;
}
