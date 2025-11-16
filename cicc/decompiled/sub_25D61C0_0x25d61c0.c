// Function: sub_25D61C0
// Address: 0x25d61c0
//
__int64 __fastcall sub_25D61C0(__int64 a1, __int64 a2, __int64 a3, __m128i a4)
{
  char v5; // dl
  __int64 v6; // r13
  void *v7; // rbx
  size_t v8; // r12
  _QWORD *v9; // rbx
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 i; // rax
  char v13; // dl
  unsigned __int64 v14; // rdi
  __int64 v15; // rsi
  unsigned int v16; // r13d
  unsigned __int64 v17; // r12
  _QWORD *v19; // r13
  int v20; // eax
  const void *v21; // r11
  __int64 v22; // r10
  __int64 v23; // rcx
  __int64 *v24; // rax
  __int64 v25; // r15
  __m128i *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 *v30; // r14
  unsigned __int8 v31; // al
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 *v34; // r14
  char v35; // al
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // [rsp+0h] [rbp-160h]
  const void *v42; // [rsp+10h] [rbp-150h]
  __int64 v43; // [rsp+18h] [rbp-148h]
  __int64 v44; // [rsp+20h] [rbp-140h]
  void *s2; // [rsp+28h] [rbp-138h]
  __int64 v47; // [rsp+40h] [rbp-120h] BYREF
  __int64 v48; // [rsp+48h] [rbp-118h] BYREF
  _QWORD *v49; // [rsp+50h] [rbp-110h] BYREF
  unsigned __int8 v50; // [rsp+58h] [rbp-108h]
  __int64 v51; // [rsp+60h] [rbp-100h] BYREF
  char v52; // [rsp+68h] [rbp-F8h]
  __int64 v53[2]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v54; // [rsp+80h] [rbp-E0h]
  __int64 v55; // [rsp+88h] [rbp-D8h]
  __int64 v56; // [rsp+90h] [rbp-D0h]
  const char *v57; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v58; // [rsp+A8h] [rbp-B8h]
  __int64 v59; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v60; // [rsp+C0h] [rbp-A0h]
  __m128i *v61; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v62; // [rsp+D8h] [rbp-88h]
  __m128i v63; // [rsp+E0h] [rbp-80h] BYREF
  __int64 *v64; // [rsp+F0h] [rbp-70h]
  __int64 v65; // [rsp+F8h] [rbp-68h]
  __m128i **v66; // [rsp+100h] [rbp-60h] BYREF
  __int64 v67[2]; // [rsp+108h] [rbp-58h] BYREF
  __int64 (__fastcall *v68)(__int64 *, __int64 *, int); // [rsp+118h] [rbp-48h]
  __int64 (__fastcall *v69)(__int64, __int64 **, __int64, __m128i); // [rsp+120h] [rbp-40h]
  char v70; // [rsp+128h] [rbp-38h]

  if ( !qword_4FF0A10 )
    sub_C64ED0("error: -function-import requires -summary-file\n", 1u);
  sub_9F1EB0((__int64)&v49, (const __m128i *)qword_4FF0A08, qword_4FF0A10, 0, a4);
  v5 = v50 & 1;
  v50 = (2 * (v50 & 1)) | v50 & 0xFD;
  if ( !v5 )
  {
    v6 = (__int64)v49;
    v53[0] = (__int64)&v61;
    v64 = (__int64 *)&v66;
    v7 = *(void **)(a1 + 168);
    v63.m128i_i64[0] = 0;
    v8 = *(_QWORD *)(a1 + 176);
    s2 = v7;
    v9 = v49 + 1;
    v49 = 0;
    v61 = 0;
    v62 = 0;
    v63.m128i_i32[2] = 0;
    v65 = 0;
    v53[1] = 0;
    v54 = 0;
    v55 = 0;
    v56 = 0;
    if ( !(_BYTE)qword_4FF0928 )
    {
      v66 = 0;
      v67[0] = 0;
      v67[1] = 0;
      LODWORD(v68) = 0;
      sub_BAF650(v6, s2, v8, (__int64)&v66);
      sub_25D6090(&v57, a2, a3, v6, 0);
      (*(void (__fastcall **)(const char *, __m128i ***, void *, size_t, __int64 *))(*(_QWORD *)v57 + 16LL))(
        v57,
        &v66,
        s2,
        v8,
        v53);
      if ( v57 )
        (*(void (__fastcall **)(const char *))(*(_QWORD *)v57 + 8LL))(v57);
      sub_C7D6A0(v67[0], 16LL * (unsigned int)v68, 8);
LABEL_7:
      v10 = *(_QWORD *)(v6 + 24);
      if ( (_QWORD *)v10 != v9 )
      {
        do
        {
          v11 = *(_QWORD *)(v10 + 64);
          for ( i = *(_QWORD *)(v10 + 56); i != v11; i += 8 )
          {
            if ( (*(_BYTE *)(*(_QWORD *)i + 12LL) & 0xFu) - 7 <= 1 )
              *(_BYTE *)(*(_QWORD *)i + 12LL) &= 0xF0u;
          }
          v10 = sub_220EEE0(v10);
        }
        while ( v9 != (_QWORD *)v10 );
      }
      goto LABEL_13;
    }
    if ( v9 == *(_QWORD **)(v6 + 24) )
    {
LABEL_13:
      sub_29DCE60(a1, v6, 0, 0);
      v66 = (__m128i **)v6;
      v68 = sub_25CCF20;
      v69 = sub_25CDBF0;
      v67[0] = a1;
      v70 = 0;
      sub_25D4030((bool *)&v51, (__int64 *)&v66, a1, v53);
      v13 = v52 & 1;
      v52 = (2 * (v52 & 1)) | v52 & 0xFD;
      if ( v13 )
      {
        v57 = "Error importing module: ";
        LOWORD(v60) = 259;
        v34 = (__int64 *)sub_CB72A0();
        v35 = v52;
        v52 &= ~2u;
        if ( (v35 & 1) != 0 )
        {
          v36 = v51;
          v51 = 0;
          v48 = 0;
          v47 = v36 | 1;
          sub_9C8CB0(&v48);
        }
        else
        {
          v47 = 1;
          v48 = 0;
          sub_9C66B0(&v48);
        }
        sub_C63F70((unsigned __int64 *)&v47, v34, v37, v38, v39, v40, (char)v57);
        sub_9C66B0(&v47);
        if ( (v52 & 2) != 0 )
          sub_A05710(&v51);
        if ( (v52 & 1) != 0 && v51 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v51 + 8LL))(v51);
      }
      if ( v68 )
        v68(v67, v67, 3);
      sub_C7D6A0(v54, 4LL * (unsigned int)v56, 4);
      if ( v64 != (__int64 *)&v66 )
        _libc_free((unsigned __int64)v64);
      sub_C7D6A0(v62, 32LL * v63.m128i_u32[2], 8);
      sub_9CD560(v6);
      v14 = v6;
      v15 = 584;
      v16 = 1;
      j_j___libc_free_0(v14);
      goto LABEL_19;
    }
    v41 = v6;
    v19 = *(_QWORD **)(v6 + 24);
    while ( 1 )
    {
      v24 = (__int64 *)v19[7];
      if ( (__int64 *)v19[8] == v24 )
        goto LABEL_28;
      v23 = *v24;
      v22 = v19[4];
      v25 = *(_QWORD *)(*v24 + 32);
      v21 = *(const void **)(*v24 + 24);
      if ( v8 == v25 )
      {
        v43 = *v24;
        v44 = v19[4];
        if ( !v8 )
          goto LABEL_28;
        v42 = *(const void **)(*v24 + 24);
        v20 = memcmp(v21, s2, v8);
        v21 = v42;
        v22 = v44;
        v23 = v43;
        if ( !v20 )
          goto LABEL_28;
      }
      if ( (*(_BYTE *)(v23 + 13) & 4) != 0 )
      {
        sub_25D0520((__int64)v53, (__int64)v21, v25, v22);
LABEL_28:
        v19 = (_QWORD *)sub_220EF30((__int64)v19);
        if ( v9 == v19 )
          goto LABEL_33;
      }
      else
      {
        sub_25D0260((__int64)v53, (__int64)v21, v25, v22);
        v19 = (_QWORD *)sub_220EF30((__int64)v19);
        if ( v9 == v19 )
        {
LABEL_33:
          v6 = v41;
          goto LABEL_7;
        }
      }
    }
  }
  v57 = (const char *)&v59;
  LOBYTE(v59) = 0;
  v58 = 0;
  sub_2240E30((__int64)&v57, qword_4FF0A10 + 20);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v58) <= 0x13
    || (sub_2241490((unsigned __int64 *)&v57, "Error loading file '", 0x14u),
        sub_2241490((unsigned __int64 *)&v57, (char *)qword_4FF0A08, qword_4FF0A10),
        (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v58) <= 2) )
  {
    sub_4262D8((__int64)"basic_string::append");
  }
  v26 = (__m128i *)sub_2241490((unsigned __int64 *)&v57, "': ", 3u);
  v61 = &v63;
  if ( (__m128i *)v26->m128i_i64[0] == &v26[1] )
  {
    v63 = _mm_loadu_si128(v26 + 1);
  }
  else
  {
    v61 = (__m128i *)v26->m128i_i64[0];
    v63.m128i_i64[0] = v26[1].m128i_i64[0];
  }
  v62 = v26->m128i_i64[1];
  v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
  v26->m128i_i64[1] = 0;
  v26[1].m128i_i8[0] = 0;
  LOWORD(v69) = 260;
  v66 = &v61;
  v30 = (__int64 *)sub_CB72A0();
  v31 = v50;
  v32 = v50 & 0xFD;
  v50 &= ~2u;
  if ( (v31 & 1) != 0 )
  {
    v33 = (__int64)v49;
    v49 = 0;
    v51 = v33 | 1;
  }
  else
  {
    v51 = 1;
    v53[0] = 0;
    sub_9C66B0(v53);
  }
  v15 = (__int64)v30;
  sub_C63F70((unsigned __int64 *)&v51, v30, v32, v27, v28, v29, (char)v66);
  if ( (v51 & 1) != 0 || (v51 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v51, (__int64)v30);
  if ( v61 != &v63 )
  {
    v15 = v63.m128i_i64[0] + 1;
    j_j___libc_free_0((unsigned __int64)v61);
  }
  if ( v57 != (const char *)&v59 )
  {
    v15 = v59 + 1;
    j_j___libc_free_0((unsigned __int64)v57);
  }
  v16 = 0;
LABEL_19:
  if ( (v50 & 2) != 0 )
    sub_25CE240(&v49, v15);
  v17 = (unsigned __int64)v49;
  if ( (v50 & 1) != 0 )
  {
    if ( v49 )
      (*(void (__fastcall **)(_QWORD *, __int64))(*v49 + 8LL))(v49, v15);
  }
  else if ( v49 )
  {
    sub_9CD560((__int64)v49);
    j_j___libc_free_0(v17);
  }
  return v16;
}
