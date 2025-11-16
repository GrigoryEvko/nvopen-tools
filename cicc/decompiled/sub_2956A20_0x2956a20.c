// Function: sub_2956A20
// Address: 0x2956a20
//
__int64 __fastcall sub_2956A20(__int64 a1, __int64 a2)
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
  __int64 v14; // rbx
  unsigned int v15; // eax
  __m128i *v16; // r12
  __int64 *i; // rbx
  __int64 v18; // rax
  __int64 *v19; // r14
  unsigned __int64 v20; // r15
  __int64 v21; // rdi
  unsigned int v22; // eax
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  __int64 v25; // rdi
  char v26; // al
  __int64 v27; // rdi
  __int64 v28; // rsi
  unsigned int v29; // r12d
  __int64 v30; // rax
  _QWORD *v31; // rbx
  _QWORD *v32; // r13
  unsigned __int64 v33; // rdi
  _QWORD *v35; // rbx
  _QWORD *v36; // r13
  unsigned __int64 v37; // rdi
  __m128i v38; // xmm1
  __int64 v39; // [rsp+8h] [rbp-128h]
  __int64 v40; // [rsp+18h] [rbp-118h]
  __int64 v41; // [rsp+20h] [rbp-110h]
  __m128i v42; // [rsp+30h] [rbp-100h] BYREF
  __m128i v43; // [rsp+40h] [rbp-F0h] BYREF
  __m128i v44; // [rsp+50h] [rbp-E0h] BYREF
  __m128i v45; // [rsp+60h] [rbp-D0h] BYREF
  __m128i v46; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v48; // [rsp+88h] [rbp-A8h]
  __int64 v49; // [rsp+90h] [rbp-A0h]
  __int64 v50; // [rsp+98h] [rbp-98h]
  __int64 (__fastcall *v51)(__int64, __int64); // [rsp+A0h] [rbp-90h]
  __m128i *v52; // [rsp+A8h] [rbp-88h]
  char v53; // [rsp+B0h] [rbp-80h]
  __int64 v54; // [rsp+B8h] [rbp-78h]
  _QWORD *v55; // [rsp+C0h] [rbp-70h]
  __int64 v56; // [rsp+C8h] [rbp-68h]
  unsigned int v57; // [rsp+D0h] [rbp-60h]
  __int64 v58; // [rsp+D8h] [rbp-58h]
  _QWORD *v59; // [rsp+E0h] [rbp-50h]
  __int64 v60; // [rsp+E8h] [rbp-48h]
  unsigned int v61; // [rsp+F0h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_68:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_68;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v7 = *(__int64 **)(a1 + 8);
  v39 = v6 + 176;
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_69:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F875EC )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_69;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F875EC);
  v11 = *(__int64 **)(a1 + 8);
  v41 = v10 + 176;
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_70:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F6D3F0 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_70;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)&v47);
  sub_983BD0((__int64)&v42, v14 + 176, a2);
  v40 = v14 + 408;
  if ( *(_BYTE *)(v14 + 488) )
  {
    *(__m128i *)(v14 + 408) = _mm_loadu_si128(&v42);
    *(__m128i *)(v14 + 424) = _mm_loadu_si128(&v43);
    *(__m128i *)(v14 + 440) = _mm_loadu_si128(&v44);
    *(__m128i *)(v14 + 456) = _mm_loadu_si128(&v45);
    *(__m128i *)(v14 + 472) = _mm_loadu_si128(&v46);
  }
  else
  {
    *(__m128i *)(v14 + 408) = _mm_loadu_si128(&v42);
    *(__m128i *)(v14 + 424) = _mm_loadu_si128(&v43);
    *(__m128i *)(v14 + 440) = _mm_loadu_si128(&v44);
    *(__m128i *)(v14 + 456) = _mm_loadu_si128(&v45);
    v38 = _mm_loadu_si128(&v46);
    *(_BYTE *)(v14 + 488) = 1;
    *(__m128i *)(v14 + 472) = v38;
  }
  sub_C7D6A0(v56, 24LL * (unsigned int)v58, 8);
  v15 = v54;
  if ( (_DWORD)v54 )
  {
    v16 = &v52[2 * (unsigned int)v54];
    for ( i = &v52->m128i_i64[1]; ; i += 4 )
    {
      v18 = *(i - 1);
      if ( v18 != -8192 && v18 != -4096 )
      {
        v19 = (__int64 *)*i;
        while ( v19 != i )
        {
          v20 = (unsigned __int64)v19;
          v19 = (__int64 *)*v19;
          v21 = *(_QWORD *)(v20 + 24);
          if ( v21 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
          j_j___libc_free_0(v20);
        }
      }
      if ( v16 == (__m128i *)(i + 3) )
        break;
    }
    v15 = v54;
  }
  sub_C7D6A0((__int64)v52, 32LL * v15, 8);
  v22 = v50;
  if ( (_DWORD)v50 )
  {
    v23 = v48;
    v24 = &v48[2 * (unsigned int)v50];
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
    v22 = v50;
  }
  sub_C7D6A0((__int64)v48, 16LL * v22, 8);
  v26 = *(_BYTE *)(a1 + 169);
  v42.m128i_i64[0] = a1;
  v54 = 0;
  v48 = (_QWORD *)v39;
  v47 = 0;
  v49 = v41;
  v53 = v26;
  v50 = v40;
  v51 = sub_2950630;
  v55 = 0;
  v52 = &v42;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  if ( (_BYTE)qword_5005988 )
  {
    v27 = 0;
    v28 = 0;
    v29 = 0;
    goto LABEL_36;
  }
  v29 = sub_2953BC0((__int64)&v47, a2);
  if ( v61 )
  {
    v35 = v59;
    v36 = &v59[6 * v61];
    while ( 1 )
    {
      while ( *v35 == -4096 )
      {
        if ( v35[1] != -4096 )
          goto LABEL_52;
        v35 += 6;
        if ( v36 == v35 )
        {
LABEL_58:
          v27 = (__int64)v59;
          v28 = 48LL * v61;
          goto LABEL_36;
        }
      }
      if ( *v35 != -8192 || v35[1] != -8192 )
      {
LABEL_52:
        v37 = v35[2];
        if ( (_QWORD *)v37 != v35 + 4 )
          _libc_free(v37);
      }
      v35 += 6;
      if ( v36 == v35 )
        goto LABEL_58;
    }
  }
  v27 = (__int64)v59;
  v28 = 0;
LABEL_36:
  sub_C7D6A0(v27, v28, 8);
  v30 = v57;
  if ( v57 )
  {
    v31 = v55;
    v32 = &v55[6 * v57];
    while ( 1 )
    {
      while ( *v31 == -4096 )
      {
        if ( v31[1] != -4096 )
          goto LABEL_39;
        v31 += 6;
        if ( v32 == v31 )
        {
LABEL_45:
          v30 = v57;
          goto LABEL_46;
        }
      }
      if ( *v31 != -8192 || v31[1] != -8192 )
      {
LABEL_39:
        v33 = v31[2];
        if ( (_QWORD *)v33 != v31 + 4 )
          _libc_free(v33);
      }
      v31 += 6;
      if ( v32 == v31 )
        goto LABEL_45;
    }
  }
LABEL_46:
  sub_C7D6A0((__int64)v55, 48 * v30, 8);
  return v29;
}
