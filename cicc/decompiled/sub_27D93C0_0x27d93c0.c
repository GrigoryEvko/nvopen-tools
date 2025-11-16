// Function: sub_27D93C0
// Address: 0x27d93c0
//
__int64 __fastcall sub_27D93C0(_QWORD *a1, __int64 a2)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __m128i v12; // xmm5
  __m128i v13; // xmm6
  __m128i v14; // xmm7
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  unsigned int v17; // eax
  _QWORD **v18; // rbx
  __int64 v19; // rax
  _QWORD *v20; // r15
  unsigned __int64 v21; // r12
  __int64 v22; // rdi
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v26; // xmm4
  unsigned int v27; // eax
  _QWORD *v28; // rbx
  _QWORD *v29; // r12
  __int64 v30; // rdi
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 *v37; // rdx
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // [rsp+0h] [rbp-100h]
  __int64 v43; // [rsp+10h] [rbp-F0h]
  _QWORD **v44; // [rsp+18h] [rbp-E8h]
  __m128i v45; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v46; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v47; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v48; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v49; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v50; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v51; // [rsp+78h] [rbp-88h]
  __int64 v52; // [rsp+80h] [rbp-80h]
  __int64 v53; // [rsp+88h] [rbp-78h]
  __int64 v54; // [rsp+90h] [rbp-70h]
  __int64 v55; // [rsp+98h] [rbp-68h]
  __int64 v56; // [rsp+A0h] [rbp-60h]
  __int64 v57; // [rsp+A8h] [rbp-58h]
  __int16 v58; // [rsp+B0h] [rbp-50h]
  __int64 v59; // [rsp+B8h] [rbp-48h]
  unsigned int v60; // [rsp+C8h] [rbp-38h]

  if ( (unsigned __int8)sub_BB98D0(a1, a2) )
    return 0;
  v4 = (__int64 *)a1[1];
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_48:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4F8144C )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_48;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F8144C);
  v8 = (__int64 *)a1[1];
  v42 = v7 + 176;
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
LABEL_49:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F6D3F0 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_49;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)&v50);
  sub_983BD0((__int64)&v45, v11 + 176, a2);
  v43 = v11 + 408;
  if ( *(_BYTE *)(v11 + 488) )
  {
    v23 = _mm_loadu_si128(&v46);
    v24 = _mm_loadu_si128(&v47);
    v25 = _mm_loadu_si128(&v48);
    v26 = _mm_loadu_si128(&v49);
    *(__m128i *)(v11 + 408) = _mm_loadu_si128(&v45);
    *(__m128i *)(v11 + 424) = v23;
    *(__m128i *)(v11 + 440) = v24;
    *(__m128i *)(v11 + 456) = v25;
    *(__m128i *)(v11 + 472) = v26;
  }
  else
  {
    v12 = _mm_loadu_si128(&v45);
    v13 = _mm_loadu_si128(&v46);
    *(_BYTE *)(v11 + 488) = 1;
    v14 = _mm_loadu_si128(&v47);
    v15 = _mm_loadu_si128(&v48);
    v16 = _mm_loadu_si128(&v49);
    *(__m128i *)(v11 + 408) = v12;
    *(__m128i *)(v11 + 424) = v13;
    *(__m128i *)(v11 + 440) = v14;
    *(__m128i *)(v11 + 456) = v15;
    *(__m128i *)(v11 + 472) = v16;
  }
  sub_C7D6A0(v59, 24LL * v60, 8);
  v17 = v57;
  if ( (_DWORD)v57 )
  {
    v18 = (_QWORD **)(v55 + 8);
    v44 = (_QWORD **)(v55 + 32LL * (unsigned int)v57);
    while ( 1 )
    {
      v19 = (__int64)*(v18 - 1);
      if ( v19 != -8192 && v19 != -4096 )
      {
        v20 = *v18;
        while ( v20 != v18 )
        {
          v21 = (unsigned __int64)v20;
          v20 = (_QWORD *)*v20;
          v22 = *(_QWORD *)(v21 + 24);
          if ( v22 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v22 + 8LL))(v22);
          j_j___libc_free_0(v21);
        }
      }
      if ( v44 == v18 + 3 )
        break;
      v18 += 4;
    }
    v17 = v57;
  }
  sub_C7D6A0(v55, 32LL * v17, 8);
  v27 = v53;
  if ( (_DWORD)v53 )
  {
    v28 = v51;
    v29 = &v51[2 * (unsigned int)v53];
    do
    {
      if ( *v28 != -8192 && *v28 != -4096 )
      {
        v30 = v28[1];
        if ( v30 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
      }
      v28 += 2;
    }
    while ( v29 != v28 );
    v27 = v53;
  }
  sub_C7D6A0((__int64)v51, 16LL * v27, 8);
  v31 = (__int64 *)a1[1];
  v32 = *v31;
  v33 = v31[1];
  if ( v32 == v33 )
LABEL_50:
    BUG();
  while ( *(_UNKNOWN **)v32 != &unk_4F8662C )
  {
    v32 += 16;
    if ( v33 == v32 )
      goto LABEL_50;
  }
  v34 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v32 + 8) + 104LL))(*(_QWORD *)(v32 + 8), &unk_4F8662C);
  v35 = sub_CFFAC0(v34, a2);
  v36 = sub_B2BEC0(a2);
  v37 = (__int64 *)a1[1];
  v38 = v36;
  v39 = *v37;
  v40 = v37[1];
  if ( v39 == v40 )
LABEL_51:
    BUG();
  while ( *(_UNKNOWN **)v39 != &unk_4F89C28 )
  {
    v39 += 16;
    if ( v40 == v39 )
      goto LABEL_51;
  }
  v41 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v39 + 8) + 104LL))(*(_QWORD *)(v39 + 8), &unk_4F89C28);
  v52 = sub_DFED00(v41, a2);
  v50 = v38;
  v53 = v42;
  v51 = (_QWORD *)v43;
  v54 = v35;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 257;
  return sub_27D8A00(a2, (__int64)&v50);
}
