// Function: sub_166F4F0
// Address: 0x166f4f0
//
_QWORD *__fastcall sub_166F4F0(
        _QWORD *a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __m128i a7)
{
  __int64 v11; // r9
  __m128i *v12; // rsi
  _BYTE *v13; // rax
  bool v14; // zf
  __m128i *v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 *v18; // r15
  __int64 *v19; // rsi
  __int64 **v20; // rax
  __int64 **v21; // rcx
  __int64 v22; // rax
  __int64 **v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  _BYTE *v27; // rsi
  __int64 v28; // rdx
  __m128i *v29; // r15
  __int64 v30; // rbx
  _BYTE *v31; // r14
  unsigned __int64 v32; // r13
  __int64 v33; // rdi
  int v35; // r9d
  __int64 v36; // rax
  __m128i *v37; // r13
  __int64 **v38; // [rsp+0h] [rbp-280h]
  __int64 **v39; // [rsp+8h] [rbp-278h]
  _BYTE *v40; // [rsp+30h] [rbp-250h] BYREF
  __int64 v41; // [rsp+38h] [rbp-248h]
  __m128i *v42; // [rsp+40h] [rbp-240h] BYREF
  __int64 v43; // [rsp+48h] [rbp-238h] BYREF
  __int64 v44; // [rsp+50h] [rbp-230h] BYREF
  __int64 v45; // [rsp+58h] [rbp-228h] BYREF
  __int64 v46; // [rsp+60h] [rbp-220h] BYREF
  __int64 v47; // [rsp+68h] [rbp-218h] BYREF
  __int64 *v48; // [rsp+70h] [rbp-210h] BYREF
  unsigned __int64 v49; // [rsp+78h] [rbp-208h] BYREF
  _QWORD v50[2]; // [rsp+80h] [rbp-200h] BYREF
  char v51; // [rsp+90h] [rbp-1F0h]
  __int64 *v52[2]; // [rsp+A0h] [rbp-1E0h] BYREF
  __int64 v53; // [rsp+B0h] [rbp-1D0h] BYREF
  __m128i *v54; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 v55; // [rsp+C8h] [rbp-1B8h]
  __m128i v56; // [rsp+D0h] [rbp-1B0h] BYREF
  _QWORD *v57; // [rsp+E0h] [rbp-1A0h] BYREF
  __m128i **v58; // [rsp+E8h] [rbp-198h]
  _QWORD *v59; // [rsp+F0h] [rbp-190h] BYREF
  __int64 v60; // [rsp+F8h] [rbp-188h]
  _QWORD v61[3]; // [rsp+100h] [rbp-180h] BYREF
  int v62; // [rsp+118h] [rbp-168h]
  __int64 v63[2]; // [rsp+120h] [rbp-160h] BYREF
  _QWORD v64[2]; // [rsp+130h] [rbp-150h] BYREF
  _QWORD *v65; // [rsp+140h] [rbp-140h]
  __int64 v66; // [rsp+148h] [rbp-138h]
  _QWORD v67[2]; // [rsp+150h] [rbp-130h] BYREF
  __int64 v68; // [rsp+160h] [rbp-120h]
  __int64 v69; // [rsp+168h] [rbp-118h]
  __int64 v70; // [rsp+170h] [rbp-110h]
  _BYTE *v71; // [rsp+178h] [rbp-108h]
  __int64 v72; // [rsp+180h] [rbp-100h]
  _BYTE v73[248]; // [rsp+188h] [rbp-F8h] BYREF

  v40 = a2;
  v41 = a3;
  LOWORD(v59) = 261;
  v57 = &v40;
  sub_16C2E90(v50, &v57, -1, 1);
  if ( (v51 & 1) != 0 && LODWORD(v50[0]) )
  {
    (*(void (__fastcall **)(__int64 **))(*(_QWORD *)v50[1] + 32LL))(v52);
    v26 = sub_2241130(v52, 0, 0, "Could not open input file: ", 27);
    v54 = &v56;
    if ( *(_QWORD *)v26 == v26 + 16 )
    {
      v56 = _mm_loadu_si128((const __m128i *)(v26 + 16));
    }
    else
    {
      v54 = *(__m128i **)v26;
      v56.m128i_i64[0] = *(_QWORD *)(v26 + 16);
    }
    v55 = *(_QWORD *)(v26 + 8);
    *(_QWORD *)v26 = v26 + 16;
    v27 = v40;
    *(_QWORD *)(v26 + 8) = 0;
    v28 = v41;
    *(_BYTE *)(v26 + 16) = 0;
    v29 = v54;
    v30 = v55;
    v57 = 0;
    v58 = 0;
    v59 = v61;
    if ( v27 )
    {
      sub_166DB50((__int64 *)&v59, v27, (__int64)&v27[v28]);
    }
    else
    {
      v60 = 0;
      LOBYTE(v61[0]) = 0;
    }
    v61[2] = -1;
    v62 = 0;
    v63[0] = (__int64)v64;
    if ( v29 )
    {
      sub_166DB50(v63, v29, (__int64)v29->m128i_i64 + v30);
    }
    else
    {
      v63[1] = 0;
      LOBYTE(v64[0]) = 0;
    }
    v66 = 0;
    v72 = 0x400000000LL;
    v65 = v67;
    LOBYTE(v67[0]) = 0;
    v68 = 0;
    v69 = 0;
    v70 = 0;
    v71 = v73;
    sub_166DE00(a4, (__int64)&v57);
    v31 = v71;
    v32 = (unsigned __int64)&v71[48 * (unsigned int)v72];
    if ( v71 != (_BYTE *)v32 )
    {
      do
      {
        v32 -= 48LL;
        v33 = *(_QWORD *)(v32 + 16);
        if ( v33 != v32 + 32 )
          j_j___libc_free_0(v33, *(_QWORD *)(v32 + 32) + 1LL);
      }
      while ( v31 != (_BYTE *)v32 );
      v32 = (unsigned __int64)v71;
    }
    if ( (_BYTE *)v32 != v73 )
      _libc_free(v32);
    if ( v68 )
      j_j___libc_free_0(v68, v70 - v68);
    if ( v65 != v67 )
      j_j___libc_free_0(v65, v67[0] + 1LL);
    if ( (_QWORD *)v63[0] != v64 )
      j_j___libc_free_0(v63[0], v64[0] + 1LL);
    if ( v59 != v61 )
      j_j___libc_free_0(v59, v61[0] + 1LL);
    if ( v54 != &v56 )
      j_j___libc_free_0(v54, v56.m128i_i64[0] + 1);
    if ( v52[0] != &v53 )
      j_j___libc_free_0(v52[0], v53 + 1);
    *a1 = 0;
    goto LABEL_52;
  }
  v12 = (__m128i *)v50[0];
  v50[0] = 0;
  v42 = v12;
  v13 = (_BYTE *)v12->m128i_i64[1];
  if ( (_BYTE *)v12[1].m128i_i64[0] == v13 )
    goto LABEL_60;
  if ( *v13 == 0xDE )
  {
    if ( v13[1] == 0xC0 && v13[2] == 23 && v13[3] == 11 )
      goto LABEL_8;
LABEL_60:
    sub_16C2FC0(&v57, v12);
    sub_38809A0((_DWORD)a1, a4, a5, 0, 1, v35, (__int64)v57, (__int64)v58, (__int64)v59, v60, (__int64)byte_3F871B3, 0);
    goto LABEL_61;
  }
  if ( *v13 != 66 || v13[1] != 67 || v13[2] != 0xC0 || v13[3] != 0xDE )
    goto LABEL_60;
LABEL_8:
  sub_1509A80((__int64)&v54, &v42, a5, a6, 0, v11, a7);
  v14 = (v55 & 1) == 0;
  v15 = v54;
  LOBYTE(v55) = v55 & 0xFD;
  if ( !v14 )
  {
    v54 = 0;
    v16 = (unsigned __int64)v15 | 1;
    v17 = (unsigned __int64)v15 & 0xFFFFFFFFFFFFFFFELL;
    v43 = v16;
    v18 = (__int64 *)v17;
    if ( v17 )
    {
      v57 = (_QWORD *)a4;
      v19 = (__int64 *)&unk_4FA032A;
      v58 = &v42;
      v43 = 0;
      v44 = 0;
      v45 = 0;
      if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v17 + 48LL))(v17, &unk_4FA032A) )
      {
        v20 = (__int64 **)v18[2];
        v21 = (__int64 **)v18[1];
        v46 = 1;
        v38 = v20;
        if ( v21 == v20 )
        {
          v24 = 1;
        }
        else
        {
          do
          {
            v39 = v21;
            v48 = *v21;
            *v21 = 0;
            sub_166EFD0((__int64 *)&v49, &v48, (__int64)&v57);
            v22 = v46;
            v19 = &v47;
            v46 = 0;
            v47 = v22 | 1;
            sub_12BEC00((unsigned __int64 *)v52, (unsigned __int64 *)&v47, &v49);
            if ( (v46 & 1) != 0 || (v23 = v39, (v46 & 0xFFFFFFFFFFFFFFFELL) != 0) )
              sub_16BCAE0(&v46);
            v46 |= (unsigned __int64)v52[0] | 1;
            if ( (v47 & 1) != 0 || (v47 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_16BCAE0(&v47);
            if ( (v49 & 1) != 0 || (v49 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_16BCAE0(&v49);
            if ( v48 )
            {
              (*(void (__fastcall **)(__int64 *))(*v48 + 8))(v48);
              v23 = v39;
            }
            v21 = v23 + 1;
          }
          while ( v38 != v21 );
          v24 = v46 | 1;
        }
        v49 = v24;
        (*(void (__fastcall **)(__int64 *))(*v18 + 8))(v18);
      }
      else
      {
        v19 = (__int64 *)v52;
        v52[0] = v18;
        sub_166EFD0((__int64 *)&v49, v52, (__int64)&v57);
        if ( v52[0] )
          (*(void (__fastcall **)(__int64 *))(*v52[0] + 8))(v52[0]);
      }
      if ( (v49 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v49 = v49 & 0xFFFFFFFFFFFFFFFELL | 1;
        sub_16BCAE0(&v49);
      }
      if ( (v45 & 1) != 0 || (v45 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_16BCAE0(&v45);
      if ( (v44 & 1) != 0 || (v44 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_16BCAE0(&v44);
      v36 = v43;
      *a1 = 0;
      if ( (v36 & 1) != 0 || (v36 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_16BCAE0(&v43);
      if ( (v55 & 2) != 0 )
        sub_1264230(&v54, (__int64)v19, v25);
      v37 = v54;
      if ( (v55 & 1) != 0 )
      {
        if ( v54 )
          (*(void (__fastcall **)(__m128i *))(v54->m128i_i64[0] + 8))(v54);
      }
      else if ( v54 )
      {
        sub_1633490(v54);
        j_j___libc_free_0(v37, 736);
      }
      goto LABEL_61;
    }
    v15 = 0;
  }
  *a1 = v15;
LABEL_61:
  if ( v42 )
    (*(void (__fastcall **)(__m128i *))(v42->m128i_i64[0] + 8))(v42);
LABEL_52:
  if ( (v51 & 1) == 0 && v50[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v50[0] + 8LL))(v50[0]);
  return a1;
}
