// Function: sub_D00390
// Address: 0xd00390
//
__int64 __fastcall sub_D00390(__int64 a1, __int64 a2)
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
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  unsigned int v19; // eax
  _QWORD **v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // r15
  _QWORD *v23; // r12
  __int64 v24; // rdi
  unsigned int v25; // eax
  _QWORD *v26; // rbx
  _QWORD *v27; // r12
  __int64 v28; // rdi
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // r12
  __m128i v33; // xmm5
  __m128i v34; // xmm6
  __m128i v35; // xmm7
  __m128i v36; // xmm0
  __m128i v37; // xmm1
  __int64 v38; // [rsp+8h] [rbp-108h]
  __int64 v39; // [rsp+10h] [rbp-100h]
  __int64 v40; // [rsp+18h] [rbp-F8h]
  __int64 v41; // [rsp+20h] [rbp-F0h]
  _QWORD **v42; // [rsp+28h] [rbp-E8h]
  __m128i v43; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v44; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v45; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v46; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v47; // [rsp+70h] [rbp-A0h] BYREF
  char v48[8]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD *v49; // [rsp+88h] [rbp-88h]
  unsigned int v50; // [rsp+98h] [rbp-78h]
  __int64 v51; // [rsp+A8h] [rbp-68h]
  unsigned int v52; // [rsp+B8h] [rbp-58h]
  __int64 v53; // [rsp+C8h] [rbp-48h]
  unsigned int v54; // [rsp+D8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_47:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8662C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_47;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8662C);
  v7 = *(__int64 **)(a1 + 8);
  v38 = v6;
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_49:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F6D3F0 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_49;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F6D3F0);
  v11 = *(__int64 **)(a1 + 8);
  v12 = v10;
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_48:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F8144C )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_48;
  }
  v39 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F8144C);
  v41 = sub_B2BEC0(a2);
  sub_BBB200((__int64)v48);
  sub_983BD0((__int64)&v43, v12 + 176, a2);
  v40 = v12 + 408;
  if ( *(_BYTE *)(v12 + 488) )
  {
    v15 = _mm_loadu_si128(&v44);
    v16 = _mm_loadu_si128(&v45);
    v17 = _mm_loadu_si128(&v46);
    v18 = _mm_loadu_si128(&v47);
    *(__m128i *)(v12 + 408) = _mm_loadu_si128(&v43);
    *(__m128i *)(v12 + 424) = v15;
    *(__m128i *)(v12 + 440) = v16;
    *(__m128i *)(v12 + 456) = v17;
    *(__m128i *)(v12 + 472) = v18;
  }
  else
  {
    v33 = _mm_loadu_si128(&v43);
    v34 = _mm_loadu_si128(&v44);
    *(_BYTE *)(v12 + 488) = 1;
    v35 = _mm_loadu_si128(&v45);
    v36 = _mm_loadu_si128(&v46);
    v37 = _mm_loadu_si128(&v47);
    *(__m128i *)(v12 + 408) = v33;
    *(__m128i *)(v12 + 424) = v34;
    *(__m128i *)(v12 + 440) = v35;
    *(__m128i *)(v12 + 456) = v36;
    *(__m128i *)(v12 + 472) = v37;
  }
  sub_C7D6A0(v53, 24LL * v54, 8);
  v19 = v52;
  if ( v52 )
  {
    v20 = (_QWORD **)(v51 + 8);
    v42 = (_QWORD **)(v51 + 32LL * v52);
    while ( 1 )
    {
      v21 = (__int64)*(v20 - 1);
      if ( v21 != -4096 && v21 != -8192 )
      {
        v22 = *v20;
        while ( v22 != v20 )
        {
          v23 = v22;
          v22 = (_QWORD *)*v22;
          v24 = v23[3];
          if ( v24 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
          j_j___libc_free_0(v23, 32);
        }
      }
      if ( v42 == v20 + 3 )
        break;
      v20 += 4;
    }
    v19 = v52;
  }
  sub_C7D6A0(v51, 32LL * v19, 8);
  v25 = v50;
  if ( v50 )
  {
    v26 = v49;
    v27 = &v49[2 * v50];
    do
    {
      if ( *v26 != -8192 && *v26 != -4096 )
      {
        v28 = v26[1];
        if ( v28 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v28 + 8LL))(v28);
      }
      v26 += 2;
    }
    while ( v27 != v26 );
    v25 = v50;
  }
  sub_C7D6A0((__int64)v49, 16LL * v25, 8);
  v29 = sub_CFFAC0(v38, a2);
  v30 = sub_22077B0(200);
  if ( v30 )
  {
    *(_QWORD *)(v30 + 8) = a2;
    *(_QWORD *)(v30 + 24) = v29;
    *(_QWORD *)v30 = v41;
    *(_QWORD *)(v30 + 32) = v39 + 176;
    *(_QWORD *)(v30 + 16) = v40;
    *(_QWORD *)(v30 + 40) = 0;
    *(_QWORD *)(v30 + 48) = v30 + 72;
    *(_QWORD *)(v30 + 56) = 16;
    *(_DWORD *)(v30 + 64) = 0;
    *(_BYTE *)(v30 + 68) = 1;
  }
  v31 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v30;
  if ( v31 )
  {
    if ( !*(_BYTE *)(v31 + 68) )
      _libc_free(*(_QWORD *)(v31 + 48), a2);
    j_j___libc_free_0(v31, 200);
  }
  return 0;
}
