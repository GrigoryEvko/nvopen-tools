// Function: sub_DA2070
// Address: 0xda2070
//
__int64 __fastcall sub_DA2070(__int64 a1, __int64 a2)
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
  _QWORD *v15; // r15
  _QWORD *v16; // r14
  __int64 v17; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  __int64 v21; // rdi
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 *v32; // rdx
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rdx
  void *v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rbx
  __int64 v39; // r12
  __m128i v41; // xmm5
  __m128i v42; // xmm6
  __m128i v43; // xmm7
  __m128i v44; // xmm0
  __m128i v45; // xmm1
  __int64 v46; // [rsp+0h] [rbp-100h]
  __int64 v47; // [rsp+8h] [rbp-F8h]
  __int64 v48; // [rsp+10h] [rbp-F0h]
  __m128i v50; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v51; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v52; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v53; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v54; // [rsp+60h] [rbp-A0h] BYREF
  _BYTE v55[8]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v56; // [rsp+78h] [rbp-88h]
  unsigned int v57; // [rsp+88h] [rbp-78h]
  __int64 v58; // [rsp+98h] [rbp-68h]
  unsigned int v59; // [rsp+A8h] [rbp-58h]
  __int64 v60; // [rsp+B8h] [rbp-48h]
  unsigned int v61; // [rsp+C8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_53:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F6D3F0 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_53;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v55);
  sub_983BD0((__int64)&v50, v6 + 176, a2);
  v48 = v6 + 408;
  if ( *(_BYTE *)(v6 + 488) )
  {
    v7 = _mm_loadu_si128(&v51);
    v8 = _mm_loadu_si128(&v52);
    v9 = _mm_loadu_si128(&v53);
    v10 = _mm_loadu_si128(&v54);
    *(__m128i *)(v6 + 408) = _mm_loadu_si128(&v50);
    *(__m128i *)(v6 + 424) = v7;
    *(__m128i *)(v6 + 440) = v8;
    *(__m128i *)(v6 + 456) = v9;
    *(__m128i *)(v6 + 472) = v10;
  }
  else
  {
    v41 = _mm_loadu_si128(&v50);
    v42 = _mm_loadu_si128(&v51);
    *(_BYTE *)(v6 + 488) = 1;
    v43 = _mm_loadu_si128(&v52);
    v44 = _mm_loadu_si128(&v53);
    v45 = _mm_loadu_si128(&v54);
    *(__m128i *)(v6 + 408) = v41;
    *(__m128i *)(v6 + 424) = v42;
    *(__m128i *)(v6 + 440) = v43;
    *(__m128i *)(v6 + 456) = v44;
    *(__m128i *)(v6 + 472) = v45;
  }
  sub_C7D6A0(v60, 24LL * v61, 8);
  v11 = v59;
  if ( v59 )
  {
    v12 = (_QWORD **)(v58 + 32LL * v59);
    for ( i = (_QWORD **)(v58 + 8); ; i += 4 )
    {
      v14 = (__int64)*(i - 1);
      if ( v14 != -8192 && v14 != -4096 )
      {
        v15 = *i;
        while ( i != v15 )
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
    v11 = v59;
  }
  sub_C7D6A0(v58, 32LL * v11, 8);
  v18 = v57;
  if ( v57 )
  {
    v19 = v56;
    v20 = &v56[2 * v57];
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
    v18 = v57;
  }
  sub_C7D6A0((__int64)v56, 16LL * v18, 8);
  v22 = *(__int64 **)(a1 + 8);
  v23 = *v22;
  v24 = v22[1];
  if ( v23 == v24 )
LABEL_50:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4F8662C )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_50;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4F8662C);
  v26 = sub_CFFAC0(v25, a2);
  v27 = *(__int64 **)(a1 + 8);
  v28 = v26;
  v29 = *v27;
  v30 = v27[1];
  if ( v29 == v30 )
LABEL_51:
    BUG();
  while ( *(_UNKNOWN **)v29 != &unk_4F8144C )
  {
    v29 += 16;
    if ( v30 == v29 )
      goto LABEL_51;
  }
  v31 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v29 + 8) + 104LL))(*(_QWORD *)(v29 + 8), &unk_4F8144C);
  v32 = *(__int64 **)(a1 + 8);
  v33 = v31 + 176;
  v34 = *v32;
  v35 = v32[1];
  if ( v34 == v35 )
LABEL_52:
    BUG();
  v36 = &unk_4F875EC;
  while ( *(_UNKNOWN **)v34 != &unk_4F875EC )
  {
    v34 += 16;
    if ( v35 == v34 )
      goto LABEL_52;
  }
  v46 = v33;
  v47 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v34 + 8) + 104LL))(*(_QWORD *)(v34 + 8), &unk_4F875EC)
      + 176;
  v37 = sub_22077B0(1576);
  v38 = v37;
  if ( v37 )
  {
    v36 = (void *)a2;
    sub_D98CB0(v37, a2, v48, v28, v46, v47);
  }
  v39 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v38;
  if ( v39 )
  {
    sub_DA11D0(v39, (__int64)v36);
    j_j___libc_free_0(v39, 1576);
  }
  return 0;
}
