// Function: sub_C59290
// Address: 0xc59290
//
_QWORD *__fastcall sub_C59290(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rsi
  unsigned int v9; // ebx
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _BYTE *v18; // rdi
  __int64 v19; // rbx
  size_t v20; // rbx
  char v21; // al
  __int64 v22; // r9
  __int64 v23; // r10
  __int64 v24; // rax
  __int64 *v25; // r13
  size_t v26; // rdx
  size_t v27; // rbx
  size_t v28; // rax
  __int64 v29; // r12
  const char *v30; // rsi
  __int64 v31; // rax
  unsigned __int64 v32; // r12
  __int64 v33; // r14
  __int64 v34; // rdi
  size_t v35; // rdx
  __int64 v36; // rax
  size_t v37; // rdi
  size_t v38; // r12
  char *v39; // r12
  size_t v40; // r14
  __int64 v41; // rdi
  unsigned __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rdi
  const void *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // rsi
  __int64 v56; // r12
  __int64 v57; // r12
  __m128i *v58; // rsi
  const void *v60; // [rsp+28h] [rbp-300h]
  __int64 v61; // [rsp+30h] [rbp-2F8h]
  void *src; // [rsp+40h] [rbp-2E8h]
  __int64 v64; // [rsp+60h] [rbp-2C8h]
  _QWORD *v65; // [rsp+68h] [rbp-2C0h] BYREF
  __int64 v66; // [rsp+70h] [rbp-2B8h]
  char v67; // [rsp+78h] [rbp-2B0h]
  _BYTE v68[32]; // [rsp+88h] [rbp-2A0h] BYREF
  _BYTE v69[32]; // [rsp+A8h] [rbp-280h] BYREF
  __m128i v70[2]; // [rsp+C8h] [rbp-260h] BYREF
  char v71; // [rsp+E8h] [rbp-240h]
  char v72; // [rsp+E9h] [rbp-23Fh]
  __m128i v73; // [rsp+F8h] [rbp-230h] BYREF
  char v74; // [rsp+108h] [rbp-220h] BYREF
  __int16 v75; // [rsp+118h] [rbp-210h]
  __m128i v76[2]; // [rsp+128h] [rbp-200h] BYREF
  __int16 v77; // [rsp+148h] [rbp-1E0h]
  __m128i v78; // [rsp+158h] [rbp-1D0h] BYREF
  __int64 v79; // [rsp+168h] [rbp-1C0h] BYREF
  __int16 v80; // [rsp+178h] [rbp-1B0h]
  __m128i v81; // [rsp+188h] [rbp-1A0h] BYREF
  char *v82; // [rsp+198h] [rbp-190h]
  size_t v83; // [rsp+1A0h] [rbp-188h]
  __int16 v84; // [rsp+1A8h] [rbp-180h]
  __m128i v85; // [rsp+1B8h] [rbp-170h] BYREF
  __int64 v86; // [rsp+1C8h] [rbp-160h]
  char v87[8]; // [rsp+1D0h] [rbp-158h] BYREF
  __int16 v88; // [rsp+1D8h] [rbp-150h]
  unsigned __int128 v89; // [rsp+258h] [rbp-D0h] BYREF
  unsigned __int64 v90; // [rsp+268h] [rbp-C0h] BYREF
  _BYTE v91[8]; // [rsp+270h] [rbp-B8h] BYREF
  __int16 v92; // [rsp+278h] [rbp-B0h]

  v8 = *(_QWORD *)(a2 + 16);
  v89 = __PAIR128__(a4, a3);
  v92 = 261;
  sub_CA4130((unsigned int)&v65, v8, (unsigned int)&v89, -1, 1, 0, 1);
  if ( (v67 & 1) != 0 )
  {
    v9 = (unsigned int)v65;
    v64 = v66;
    (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)v66 + 32LL))(v68, v66, (unsigned int)v65);
    v73.m128i_i64[0] = a3;
    v78.m128i_i64[0] = (__int64)"': ";
    v73.m128i_i64[1] = a4;
    v88 = 260;
    v75 = 261;
    v70[0].m128i_i64[0] = (__int64)"cannot not open file '";
    v85.m128i_i64[0] = (__int64)v68;
    v80 = 259;
    v72 = 1;
    v71 = 3;
    sub_9C6370(v76, v70, &v73, v10, 260, 261);
    sub_9C6370(&v81, v76, &v78, v11, v12, v13);
    sub_9C6370((__m128i *)&v89, &v81, &v85, v14, v15, v16);
    sub_CA0F50(v69, &v89);
    sub_C63F00(a1, v69, v9, v64);
    sub_2240A30(v69);
    sub_2240A30(v68);
    if ( (v67 & 1) == 0 )
      goto LABEL_45;
    return a1;
  }
  v18 = (_BYTE *)v65[1];
  v19 = v65[2];
  v73.m128i_i64[1] = 0;
  v20 = v19 - (_QWORD)v18;
  v73.m128i_i64[0] = (__int64)&v74;
  v74 = 0;
  v21 = sub_C5E6F0(v18, v20);
  v22 = v20;
  v23 = (__int64)v18;
  if ( v21 )
  {
    if ( !(unsigned __int8)sub_C5E730(v18, v20) )
    {
      v56 = sub_2241E50(v18, v20, v49, v50, v51);
      *(_QWORD *)&v89 = &v90;
      sub_C4FB50((__int64 *)&v89, "Could not convert UTF16 to UTF8", (__int64)"");
      sub_C63F00(a1, &v89, 84, v56);
      sub_2240A30(&v89);
      goto LABEL_44;
    }
    v22 = v73.m128i_i64[1];
    v23 = v73.m128i_i64[0];
  }
  else if ( v20 > 2 && *v18 == 0xEF && v18[1] == 0xBB && v18[2] == 0xBF )
  {
    v23 = (__int64)(v18 + 3);
    v22 = v20 - 3;
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(a2 + 8))(
    v23,
    v22,
    a2,
    a5,
    *(unsigned __int8 *)(a2 + 57));
  if ( !*(_BYTE *)(a2 + 56) && !*(_BYTE *)(a2 + 58)
    || (v24 = sub_C80DA0(a3, a4, 0),
        v25 = *(__int64 **)a5,
        src = (void *)v24,
        v27 = v26,
        v61 = *(_QWORD *)a5 + 8LL * *(unsigned int *)(a5 + 8),
        v61 == *(_QWORD *)a5) )
  {
LABEL_43:
    *(_QWORD *)&v89 = 0;
    *a1 = 1;
    sub_9C66B0((__int64 *)&v89);
    goto LABEL_44;
  }
  while ( 1 )
  {
    while ( 1 )
    {
      v29 = *v25;
      if ( !*v25 )
        goto LABEL_14;
      if ( *(_BYTE *)(a2 + 58) )
      {
        v70[0].m128i_i64[0] = *v25;
        v30 = "<CFGDIR>";
        v70[0].m128i_i64[1] = strlen((const char *)v70[0].m128i_i64[0]);
        *(_QWORD *)&v89 = v91;
        *((_QWORD *)&v89 + 1) = 0;
        v90 = 128;
        v31 = sub_C931B0(v70, "<CFGDIR>", 8, 0);
        if ( v31 != -1 )
        {
          v32 = 0;
          v33 = v31;
          while ( 1 )
          {
            v36 = v70[0].m128i_i64[1];
            if ( v32 <= v70[0].m128i_i64[1] )
              v36 = v32;
            v37 = v70[0].m128i_i64[1] - v36;
            if ( v33 - v32 <= v70[0].m128i_i64[1] - v36 )
              v37 = v33 - v32;
            v45 = (const void *)(v70[0].m128i_i64[0] + v36);
            v38 = v37;
            v44 = *((_QWORD *)&v89 + 1);
            if ( !*((_QWORD *)&v89 + 1) )
              break;
            v88 = 257;
            v84 = 257;
            v80 = 257;
            v77 = 261;
            v76[0].m128i_i64[0] = (__int64)v45;
            v76[0].m128i_i64[1] = v38;
            sub_C81B70(&v89, v76, &v78, &v81, &v85);
            v34 = *((_QWORD *)&v89 + 1);
            v35 = *((_QWORD *)&v89 + 1) + v27;
            if ( *((_QWORD *)&v89 + 1) + v27 > v90 )
              goto LABEL_33;
LABEL_24:
            if ( v27 )
            {
              memcpy((void *)(v89 + v34), src, v27);
              v34 = *((_QWORD *)&v89 + 1);
            }
            v32 = v33 + 8;
            *((_QWORD *)&v89 + 1) = v27 + v34;
            v30 = "<CFGDIR>";
            v33 = sub_C931B0(v70, "<CFGDIR>", 8, v33 + 8);
            if ( v33 == -1 )
            {
              v46 = *((_QWORD *)&v89 + 1);
              if ( *((_QWORD *)&v89 + 1) )
              {
                if ( v32 <= v70[0].m128i_i64[1] )
                {
                  v47 = v70[0].m128i_i64[1] - v32;
                  v48 = v32 + v70[0].m128i_i64[0];
LABEL_50:
                  if ( v47 == -1 || v47 )
                  {
                    v76[0].m128i_i64[0] = v48;
                    v88 = 257;
                    v84 = 257;
                    v80 = 257;
                    v77 = 261;
                    v76[0].m128i_i64[1] = v47;
                    sub_C81B70(&v89, v76, &v78, &v81, &v85);
                  }
                  v46 = *((_QWORD *)&v89 + 1);
                }
                v30 = (const char *)v89;
                *v25 = sub_C948A0(a2, v89, v46);
              }
              goto LABEL_54;
            }
          }
          if ( v90 < v38 )
          {
            v60 = v45;
            sub_C8D290(&v89, v91, v38, 1);
            v44 = *((_QWORD *)&v89 + 1);
            v45 = v60;
          }
          if ( v38 )
          {
            memcpy((void *)(v89 + v44), v45, v38);
            v44 = *((_QWORD *)&v89 + 1);
          }
          v34 = v38 + v44;
          *((_QWORD *)&v89 + 1) = v34;
          v35 = v34 + v27;
          if ( v34 + v27 <= v90 )
            goto LABEL_24;
LABEL_33:
          sub_C8D290(&v89, v91, v35, 1);
          v34 = *((_QWORD *)&v89 + 1);
          goto LABEL_24;
        }
        if ( *((_QWORD *)&v89 + 1) )
        {
          v47 = v70[0].m128i_i64[1];
          v48 = v70[0].m128i_i64[0];
          goto LABEL_50;
        }
LABEL_54:
        if ( (_BYTE *)v89 != v91 )
          _libc_free(v89, v30);
        v29 = *v25;
        if ( !*v25 )
          goto LABEL_14;
      }
      v28 = strlen((const char *)v29);
      if ( !v28 )
        goto LABEL_14;
      if ( *(_BYTE *)v29 != 64 )
        break;
      v39 = (char *)(v29 + 1);
      v40 = v28 - 1;
      v92 = 261;
      *(_QWORD *)&v89 = v39;
      *((_QWORD *)&v89 + 1) = v28 - 1;
      if ( (unsigned __int8)sub_C81F30(&v89, 0) )
      {
        v86 = 128;
        v41 = 1;
        v87[0] = 64;
        v85.m128i_i64[1] = 1;
        v85.m128i_i64[0] = (__int64)v87;
        v42 = 128;
LABEL_36:
        if ( v42 < v41 + v27 )
        {
          sub_C8D290(&v85, v87, v41 + v27, 1);
          v41 = v85.m128i_i64[1];
        }
        if ( v27 )
        {
          memcpy((void *)(v85.m128i_i64[0] + v41), src, v27);
          v41 = v85.m128i_i64[1];
        }
        v85.m128i_i64[1] = v27 + v41;
        v92 = 257;
        v84 = 257;
        v80 = 257;
        v77 = 261;
        v76[0].m128i_i64[0] = (__int64)v39;
        v76[0].m128i_i64[1] = v40;
        sub_C81B70(&v85, v76, &v78, &v81, &v89);
        goto LABEL_41;
      }
LABEL_14:
      if ( (__int64 *)v61 == ++v25 )
        goto LABEL_43;
    }
    if ( v28 <= 8 || *(_QWORD *)v29 != 0x6769666E6F632D2DLL || *(_BYTE *)(v29 + 8) != 61 )
      goto LABEL_14;
    v40 = v28 - 9;
    v39 = (char *)(v29 + 9);
    v85.m128i_i64[0] = (__int64)v87;
    v86 = 128;
    v87[0] = 64;
    v85.m128i_i64[1] = 1;
    v92 = 261;
    *(_QWORD *)&v89 = v39;
    *((_QWORD *)&v89 + 1) = v28 - 9;
    if ( (unsigned __int8)sub_C81CA0(&v89, 0) )
    {
      v41 = v85.m128i_i64[1];
      v42 = v86;
      goto LABEL_36;
    }
    *(_QWORD *)&v89 = v91;
    *((_QWORD *)&v89 + 1) = 0;
    v90 = 128;
    if ( !(unsigned __int8)sub_C58D10((_QWORD *)a2, v39, v40, &v89) )
      break;
    v55 = v89;
    sub_C58CA0(&v85, (_BYTE *)v89, (_BYTE *)(v89 + *((_QWORD *)&v89 + 1)));
    if ( (_BYTE *)v89 != v91 )
      _libc_free(v89, v55);
LABEL_41:
    v43 = v85.m128i_i64[0];
    *v25 = sub_C948A0(a2, v85.m128i_i64[0], v85.m128i_i64[1]);
    if ( (char *)v85.m128i_i64[0] == v87 )
      goto LABEL_14;
    _libc_free(v85.m128i_i64[0], v43);
    if ( (__int64 *)v61 == ++v25 )
      goto LABEL_43;
  }
  v82 = v39;
  v84 = 1283;
  v81.m128i_i64[0] = (__int64)"cannot not find configuration file: ";
  v83 = v40;
  v57 = sub_2241E50(a2, v39, v52, v53, v54);
  sub_CA0F50(&v78, &v81);
  v58 = &v78;
  sub_C63F00(a1, &v78, 2, v57);
  if ( (__int64 *)v78.m128i_i64[0] != &v79 )
  {
    v58 = (__m128i *)(v79 + 1);
    j_j___libc_free_0(v78.m128i_i64[0], v79 + 1);
  }
  if ( (_BYTE *)v89 != v91 )
    _libc_free(v89, v58);
  if ( (char *)v85.m128i_i64[0] != v87 )
    _libc_free(v85.m128i_i64[0], v58);
LABEL_44:
  sub_2240A30(&v73);
  if ( (v67 & 1) == 0 )
  {
LABEL_45:
    if ( v65 )
      (*(void (__fastcall **)(_QWORD *))(*v65 + 8LL))(v65);
  }
  return a1;
}
