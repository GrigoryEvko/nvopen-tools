// Function: sub_2612830
// Address: 0x2612830
//
__int64 __fastcall sub_2612830(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  int v6; // eax
  __int128 v7; // xmm0
  __int64 v8; // rcx
  __int64 v9; // rsi
  int v10; // edi
  __int128 v11; // xmm1
  __int128 v12; // xmm2
  __int128 v13; // xmm3
  __int128 v14; // xmm4
  int v15; // eax
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rdx
  _QWORD *v19; // rbx
  _QWORD *v20; // rax
  _QWORD *v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rbx
  _QWORD *v24; // rax
  __int64 v25; // r9
  _QWORD *v26; // rdi
  char *v27; // rsi
  _QWORD *v28; // rax
  _QWORD *v29; // r14
  _QWORD *v30; // rax
  _QWORD *v31; // rdi
  char *v32; // rsi
  _QWORD *v33; // r14
  unsigned __int64 *v34; // r12
  _QWORD *v35; // rbx
  char *v36; // rsi
  _QWORD *v37; // rdx
  __int64 v38; // rdi
  _QWORD *v40; // r12
  __int64 v41; // rdx
  _QWORD *v42; // r14
  __int64 v43; // [rsp+60h] [rbp-D0h]
  _QWORD *v44; // [rsp+68h] [rbp-C8h]
  int v45; // [rsp+70h] [rbp-C0h]
  __int64 v46; // [rsp+70h] [rbp-C0h]
  __int64 v47; // [rsp+70h] [rbp-C0h]
  unsigned __int64 v49; // [rsp+80h] [rbp-B0h]
  __int64 v50; // [rsp+80h] [rbp-B0h]
  __int64 v51; // [rsp+80h] [rbp-B0h]
  __int64 v52; // [rsp+80h] [rbp-B0h]
  __int64 v53; // [rsp+80h] [rbp-B0h]
  void *v55; // [rsp+90h] [rbp-A0h]
  void *v56; // [rsp+98h] [rbp-98h]
  __int128 v57; // [rsp+A0h] [rbp-90h] BYREF
  unsigned __int64 v58; // [rsp+B0h] [rbp-80h]
  int v59; // [rsp+B8h] [rbp-78h]
  char v60; // [rsp+BCh] [rbp-74h]
  _QWORD v61[2]; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v62; // [rsp+D0h] [rbp-60h] BYREF
  _QWORD *v63; // [rsp+D8h] [rbp-58h]
  __int64 v64; // [rsp+E0h] [rbp-50h]
  int v65; // [rsp+E8h] [rbp-48h]
  char v66; // [rsp+ECh] [rbp-44h]
  _QWORD v67[8]; // [rsp+F0h] [rbp-40h] BYREF

  v6 = sub_BC0510(a4, &unk_502F110, (__int64)a3);
  v7 = (__int128)_mm_loadu_si128((const __m128i *)a2);
  v8 = *(_QWORD *)(a2 + 84);
  v9 = *(unsigned int *)(a2 + 92);
  v10 = v6 + 8;
  v11 = (__int128)_mm_loadu_si128((const __m128i *)(a2 + 16));
  v57 = xmmword_4FF2748;
  v12 = (__int128)_mm_loadu_si128((const __m128i *)(a2 + 32));
  v13 = (__int128)_mm_loadu_si128((const __m128i *)(a2 + 48));
  v58 = __PAIR64__(dword_4FF2288, dword_4FF24E8);
  v14 = (__int128)_mm_loadu_si128((const __m128i *)(a2 + 64));
  v15 = *(_DWORD *)(a2 + 80);
  v59 = dword_4FF2028;
  v56 = (void *)(a1 + 32);
  v55 = (void *)(a1 + 80);
  if ( !(unsigned __int8)sub_30CC0B0(v10, v9, (unsigned int)&v57, v8, v16, v17, v7, v11, v12, v13, v14, v15) )
  {
    LOWORD(v61[0]) = 259;
    v38 = *a3;
    *(_QWORD *)&v57 = "Could not setup Inlining Advisor for the requested mode and/or options";
    sub_B6ECE0(v38, (__int64)&v57);
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_QWORD *)(a1 + 8) = v56;
    *(_DWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 56) = v55;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v18 = *(_QWORD *)(a2 + 120);
  v49 = *(_QWORD *)(a2 + 104);
  v19 = *(_QWORD **)(a2 + 112);
  v45 = *(_DWORD *)(a2 + 96);
  if ( !v45 )
  {
    *(_QWORD *)(a2 + 120) = 0;
    *(_QWORD *)(a2 + 112) = 0;
    *(_QWORD *)(a2 + 104) = 0;
    v46 = v18;
    v28 = (_QWORD *)sub_22077B0(0x30u);
    v29 = v28;
    if ( v28 )
    {
      v28[2] = v19;
      v28[3] = v46;
      v28[4] = 0;
      *v28 = &unk_4A0C3B8;
      v28[5] = 0;
      v28[1] = v49;
    }
    else
    {
      if ( (_QWORD *)v49 != v19 )
      {
        v47 = a1;
        v40 = (_QWORD *)v49;
        do
        {
          if ( *v40 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v40 + 8LL))(*v40);
          ++v40;
        }
        while ( v40 != v19 );
        a1 = v47;
      }
      if ( v49 )
        j_j___libc_free_0(v49);
    }
    v23 = a2 + 144;
    v30 = (_QWORD *)sub_22077B0(0x10u);
    v31 = v30;
    if ( v30 )
    {
      v30[1] = v29;
      v29 = 0;
      *v30 = &unk_4A0C3F8;
    }
    *(_QWORD *)&v57 = v30;
    v32 = *(char **)(a2 + 152);
    if ( v32 == *(char **)(a2 + 160) )
    {
      sub_2275C60((unsigned __int64 *)(a2 + 144), v32, &v57);
      v31 = (_QWORD *)v57;
    }
    else
    {
      if ( v32 )
      {
        *(_QWORD *)v32 = v30;
        *(_QWORD *)(a2 + 152) += 8LL;
LABEL_20:
        if ( v29 )
          (*(void (__fastcall **)(_QWORD *))(*v29 + 8LL))(v29);
        goto LABEL_22;
      }
      *(_QWORD *)(a2 + 152) = 8;
    }
    if ( v31 )
      (*(void (__fastcall **)(_QWORD *))(*v31 + 8LL))(v31);
    goto LABEL_20;
  }
  *(_QWORD *)(a2 + 120) = 0;
  *(_QWORD *)(a2 + 112) = 0;
  *(_QWORD *)(a2 + 104) = 0;
  v43 = v18;
  v20 = (_QWORD *)sub_22077B0(0x30u);
  v44 = v20;
  v21 = v20;
  if ( v20 )
  {
    v20[2] = v19;
    v20[3] = v43;
    v20[4] = 0;
    *v20 = &unk_4A0C3B8;
    v20[5] = 0;
    v20[1] = v49;
    v22 = sub_22077B0(0x18u);
    if ( v22 )
    {
LABEL_5:
      *(_QWORD *)v22 = &unk_4A12578;
      *(_QWORD *)(v22 + 8) = v44;
      *(_DWORD *)(v22 + 16) = v45;
      goto LABEL_6;
    }
    (*(void (__fastcall **)(_QWORD *))(*v44 + 8LL))(v44);
    v22 = 0;
  }
  else
  {
    v41 = v43 - v49;
    if ( (_QWORD *)v49 != v19 )
    {
      v42 = (_QWORD *)v49;
      do
      {
        if ( *v42 )
          (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD *))(*(_QWORD *)*v42 + 8LL))(*v42, v9, v41, v21);
        ++v42;
      }
      while ( v42 != v19 );
    }
    if ( v49 )
      j_j___libc_free_0(v49);
    v22 = sub_22077B0(0x18u);
    if ( v22 )
      goto LABEL_5;
  }
LABEL_6:
  v50 = v22;
  v23 = a2 + 144;
  v24 = (_QWORD *)sub_22077B0(0x10u);
  v25 = v50;
  v26 = v24;
  if ( v24 )
  {
    v24[1] = v50;
    v25 = 0;
    *v24 = &unk_4A0C3F8;
  }
  *(_QWORD *)&v57 = v24;
  v27 = *(char **)(a2 + 152);
  if ( v27 == *(char **)(a2 + 160) )
  {
    v53 = v25;
    sub_2275C60((unsigned __int64 *)(a2 + 144), v27, &v57);
    v26 = (_QWORD *)v57;
    v25 = v53;
LABEL_63:
    if ( v26 )
    {
      v52 = v25;
      (*(void (__fastcall **)(_QWORD *))(*v26 + 8LL))(v26);
      v25 = v52;
    }
    goto LABEL_11;
  }
  if ( !v27 )
  {
    *(_QWORD *)(a2 + 152) = 8;
    goto LABEL_63;
  }
  *(_QWORD *)v27 = v24;
  *(_QWORD *)(a2 + 152) += 8LL;
LABEL_11:
  if ( v25 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
LABEL_22:
  v33 = *(_QWORD **)(a2 + 184);
  if ( v33 != *(_QWORD **)(a2 + 192) )
  {
    v51 = a1;
    v34 = (unsigned __int64 *)v23;
    v35 = *(_QWORD **)(a2 + 192);
    do
    {
      while ( 1 )
      {
        v36 = *(char **)(a2 + 152);
        if ( v36 != *(char **)(a2 + 160) )
          break;
        v37 = v33++;
        sub_2275C60(v34, v36, v37);
        if ( v35 == v33 )
          goto LABEL_29;
      }
      if ( v36 )
      {
        *(_QWORD *)v36 = *v33;
        *v33 = 0;
        v36 = *(char **)(a2 + 152);
      }
      ++v33;
      *(_QWORD *)(a2 + 152) = v36 + 8;
    }
    while ( v35 != v33 );
LABEL_29:
    v23 = (__int64)v34;
    a1 = v51;
  }
  sub_BC0DB0((__int64)&v57, v23, (__int64)a3, a4);
  if ( !v66 )
    _libc_free((unsigned __int64)v63);
  if ( !v60 )
    _libc_free(*((unsigned __int64 *)&v57 + 1));
  v58 = 0x100000002LL;
  *((_QWORD *)&v57 + 1) = v61;
  v62 = 0;
  v63 = v67;
  v64 = 2;
  v65 = 0;
  v66 = 1;
  v59 = 0;
  v60 = 1;
  v61[0] = &qword_4F82400;
  *(_QWORD *)&v57 = 1;
  if ( !(_BYTE)qword_4FF2928 )
  {
    if ( &qword_4F82400 == (__int64 *)&unk_502F110 )
    {
      HIDWORD(v58) = 0;
      *(_QWORD *)&v57 = 2;
    }
    HIDWORD(v64) = 1;
    v62 = 1;
    v67[0] = &unk_502F110;
  }
  sub_C8CF70(a1, v56, 2, (__int64)v61, (__int64)&v57);
  sub_C8CF70(a1 + 48, v55, 2, (__int64)v67, (__int64)&v62);
  if ( !v66 )
    _libc_free((unsigned __int64)v63);
  if ( !v60 )
    _libc_free(*((unsigned __int64 *)&v57 + 1));
  return a1;
}
