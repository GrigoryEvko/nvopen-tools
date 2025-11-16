// Function: sub_1027E50
// Address: 0x1027e50
//
__int64 __fastcall sub_1027E50(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // xmm4
  unsigned int v15; // eax
  _QWORD **v16; // r12
  _QWORD **i; // rbx
  __int64 v18; // rax
  _QWORD *v19; // r15
  _QWORD *v20; // r14
  __int64 v21; // rdi
  unsigned int v22; // eax
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  __int64 v25; // rdi
  _QWORD *v26; // rax
  _QWORD *v27; // rdx
  _QWORD *v28; // rcx
  _QWORD *v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // rdi
  __m128i v33; // xmm5
  __m128i v34; // xmm6
  __m128i v35; // xmm7
  __m128i v36; // xmm0
  __m128i v37; // xmm1
  __int64 v38; // [rsp+8h] [rbp-F8h]
  __int64 v39; // [rsp+10h] [rbp-F0h]
  __m128i v40; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v41; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v42; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v43; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v44; // [rsp+60h] [rbp-A0h] BYREF
  char v45[8]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v46; // [rsp+78h] [rbp-88h]
  unsigned int v47; // [rsp+88h] [rbp-78h]
  __int64 v48; // [rsp+98h] [rbp-68h]
  unsigned int v49; // [rsp+A8h] [rbp-58h]
  __int64 v50; // [rsp+B8h] [rbp-48h]
  unsigned int v51; // [rsp+C8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_48:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F875EC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_48;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F875EC);
  v7 = *(__int64 **)(a1 + 8);
  v38 = v6 + 176;
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_47:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F6D3F0 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_47;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v45);
  sub_983BD0((__int64)&v40, v10 + 176, a2);
  v39 = v10 + 408;
  if ( *(_BYTE *)(v10 + 488) )
  {
    v11 = _mm_loadu_si128(&v41);
    v12 = _mm_loadu_si128(&v42);
    v13 = _mm_loadu_si128(&v43);
    v14 = _mm_loadu_si128(&v44);
    *(__m128i *)(v10 + 408) = _mm_loadu_si128(&v40);
    *(__m128i *)(v10 + 424) = v11;
    *(__m128i *)(v10 + 440) = v12;
    *(__m128i *)(v10 + 456) = v13;
    *(__m128i *)(v10 + 472) = v14;
  }
  else
  {
    v33 = _mm_loadu_si128(&v40);
    v34 = _mm_loadu_si128(&v41);
    *(_BYTE *)(v10 + 488) = 1;
    v35 = _mm_loadu_si128(&v42);
    v36 = _mm_loadu_si128(&v43);
    v37 = _mm_loadu_si128(&v44);
    *(__m128i *)(v10 + 408) = v33;
    *(__m128i *)(v10 + 424) = v34;
    *(__m128i *)(v10 + 440) = v35;
    *(__m128i *)(v10 + 456) = v36;
    *(__m128i *)(v10 + 472) = v37;
  }
  sub_C7D6A0(v50, 24LL * v51, 8);
  v15 = v49;
  if ( v49 )
  {
    v16 = (_QWORD **)(v48 + 32LL * v49);
    for ( i = (_QWORD **)(v48 + 8); ; i += 4 )
    {
      v18 = (__int64)*(i - 1);
      if ( v18 != -8192 && v18 != -4096 )
      {
        v19 = *i;
        while ( v19 != i )
        {
          v20 = v19;
          v19 = (_QWORD *)*v19;
          v21 = v20[3];
          if ( v21 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
          j_j___libc_free_0(v20, 32);
        }
      }
      if ( v16 == i + 3 )
        break;
    }
    v15 = v49;
  }
  sub_C7D6A0(v48, 32LL * v15, 8);
  v22 = v47;
  if ( v47 )
  {
    v23 = v46;
    v24 = &v46[2 * v47];
    do
    {
      if ( *v23 != -8192 && *v23 != -4096 )
      {
        v25 = v23[1];
        if ( v25 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
      }
      v23 += 2;
    }
    while ( v24 != v23 );
    v22 = v47;
  }
  sub_C7D6A0((__int64)v46, 16LL * v22, 8);
  v26 = (_QWORD *)sub_22077B0(312);
  v27 = v26;
  if ( v26 )
  {
    *v26 = 0;
    v28 = v26 + 21;
    v29 = v26 + 13;
    *(v29 - 12) = 0;
    *(v29 - 11) = 0;
    *((_DWORD *)v29 - 20) = 0;
    *(v29 - 9) = 0;
    *(v29 - 8) = 0;
    *(v29 - 7) = 0;
    *((_DWORD *)v29 - 12) = 0;
    *(v29 - 5) = 0;
    *(v29 - 4) = 0;
    *(v29 - 3) = 0;
    *(v29 - 2) = 0;
    *(v29 - 1) = 1;
    do
    {
      if ( v29 )
        *v29 = -4096;
      v29 += 2;
    }
    while ( v29 != v28 );
    v30 = v27 + 23;
    v27[21] = 0;
    v27[22] = 1;
    do
    {
      if ( v30 )
      {
        *v30 = -4096;
        *((_DWORD *)v30 + 2) = 0x7FFFFFFF;
      }
      v30 += 3;
    }
    while ( v30 != v27 + 35 );
    *((_BYTE *)v27 + 280) = 0;
    v27[36] = a2;
    v27[37] = v38;
    v27[38] = v39;
  }
  v31 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v27;
  if ( v31 )
    sub_1027BE0(v31);
  return 0;
}
