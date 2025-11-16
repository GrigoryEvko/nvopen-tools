// Function: sub_C894D0
// Address: 0xc894d0
//
__m128i *__fastcall sub_C894D0(
        __m128i *a1,
        _QWORD *a2,
        char *a3,
        unsigned __int64 a4,
        char *a5,
        __int64 a6,
        _QWORD *a7)
{
  char *v9; // rsi
  __int64 v11; // rdx
  __int64 *v12; // rcx
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r13
  char *v17; // rsi
  size_t v18; // rcx
  char *v19; // r13
  unsigned __int64 v20; // r12
  char v21; // r12
  __int64 v22; // r13
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  size_t v25; // r9
  unsigned __int64 v26; // rax
  char *v27; // rdx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // r12
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // r13
  _BYTE *v34; // rax
  unsigned __int64 v35; // rax
  size_t v36; // r8
  bool v37; // cc
  size_t v38; // rax
  size_t v39; // r12
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rax
  char *v43; // r13
  unsigned __int64 v44; // rdx
  char *v45; // rdi
  unsigned __int64 v46; // r12
  __int64 *v47; // rdx
  __int64 v48; // rsi
  unsigned __int64 v49; // rdx
  _BYTE *v50; // rdi
  size_t v51; // rsi
  __int64 v52; // rdx
  _QWORD *v53; // rdi
  char *v54; // r13
  unsigned __int64 v55; // rsi
  _BYTE *v56; // rdi
  _BYTE *v57; // rax
  size_t v58; // rsi
  __int64 v59; // rdi
  size_t v60; // rdx
  size_t v61; // rdx
  size_t n; // [rsp+28h] [rbp-178h]
  size_t nb; // [rsp+28h] [rbp-178h]
  size_t na; // [rsp+28h] [rbp-178h]
  char *v68; // [rsp+30h] [rbp-170h] BYREF
  unsigned __int64 v69; // [rsp+38h] [rbp-168h]
  __m128i *v70; // [rsp+40h] [rbp-160h] BYREF
  __int64 v71; // [rsp+48h] [rbp-158h]
  __m128i v72; // [rsp+50h] [rbp-150h] BYREF
  _QWORD *v73; // [rsp+60h] [rbp-140h] BYREF
  size_t v74; // [rsp+68h] [rbp-138h]
  _QWORD src[2]; // [rsp+70h] [rbp-130h] BYREF
  _QWORD v76[2]; // [rsp+80h] [rbp-120h] BYREF
  char *v77; // [rsp+90h] [rbp-110h]
  unsigned __int64 v78; // [rsp+98h] [rbp-108h]
  __int16 v79; // [rsp+A0h] [rbp-100h]
  unsigned int v80[4]; // [rsp+B0h] [rbp-F0h] BYREF
  char *v81; // [rsp+C0h] [rbp-E0h]
  __int16 v82; // [rsp+D0h] [rbp-D0h]
  _BYTE *v83; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v84; // [rsp+E8h] [rbp-B8h]
  _BYTE v85[176]; // [rsp+F0h] [rbp-B0h] BYREF

  v68 = a3;
  v69 = a4;
  v83 = v85;
  v84 = 0x800000000LL;
  if ( !(unsigned __int8)sub_C89090(a2, a5, a6, (__int64)&v83, a7) )
  {
    v9 = a5;
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_C88E90(a1->m128i_i64, a5, (__int64)&a5[a6]);
    goto LABEL_3;
  }
  v11 = *(_QWORD *)v83;
  v70 = &v72;
  sub_C88E90((__int64 *)&v70, a5, v11);
  if ( !v69 )
    goto LABEL_24;
  while ( 1 )
  {
    LOBYTE(v80[0]) = 92;
    v13 = sub_C931B0(&v68, v80, 1, 0);
    if ( v13 == -1 )
      break;
    v15 = v69;
    v16 = v13 + 1;
    v17 = v68;
    if ( v13 + 1 > v69 )
    {
      if ( v13 <= v69 )
        v15 = v13;
      v20 = v15;
      if ( 0x3FFFFFFFFFFFFFFFLL - v71 >= v15 )
        goto LABEL_50;
LABEL_46:
      sub_4262D8((__int64)"basic_string::append");
    }
    v18 = v69 - v16;
    v19 = &v68[v16];
    if ( v13 <= v69 )
      v15 = v13;
    v20 = v15;
    if ( 0x3FFFFFFFFFFFFFFFLL - v71 < v15 )
      goto LABEL_46;
    n = v18;
    sub_2241490(&v70, v68, v15, v18);
    v12 = (__int64 *)n;
    if ( !n )
      goto LABEL_51;
    v68 = v19;
    v69 = n;
    v21 = *v19;
    if ( *v19 == 110 )
    {
      v30 = v71;
      v40 = (__int64)v70;
      v41 = 15;
      if ( v70 != &v72 )
        v41 = v72.m128i_i64[0];
      v33 = v71 + 1;
      if ( v71 + 1 > v41 )
      {
        sub_2240BB0(&v70, v71, 0, 0, 1);
        v40 = (__int64)v70;
      }
      *(_BYTE *)(v40 + v30) = 10;
    }
    else
    {
      if ( v21 <= 110 )
      {
        if ( v21 <= 57 )
        {
          if ( v21 > 47 )
          {
            v42 = sub_C935B0(&v68, "0123456789", 10, 0);
            v43 = v68;
            v44 = v42;
            v45 = v68;
            if ( v42 > v69 )
              v44 = v69;
            v46 = v44;
            v68 += v44;
            v69 -= v44;
            if ( !(unsigned __int8)sub_C93C90(v45, v44, 10, v80)
              && *(_QWORD *)v80 == v80[0]
              && (unsigned int)v84 > v80[0] )
            {
              v47 = (__int64 *)&v83[16 * v80[0]];
              v48 = *v47;
              v49 = v47[1];
              if ( v49 > 0x3FFFFFFFFFFFFFFFLL - v71 )
                goto LABEL_46;
LABEL_69:
              sub_2241490(&v70, v48, v49, v12);
              v28 = v69;
              goto LABEL_23;
            }
            if ( !a7 || a7[1] )
              goto LABEL_66;
            v78 = v46;
            v76[0] = "invalid backreference string '";
            v79 = 1283;
            *(_QWORD *)v80 = v76;
            v81 = "'";
            v77 = v43;
            v82 = 770;
            sub_CA0F50(&v73, v80);
            v50 = (_BYTE *)*a7;
            if ( v73 == src )
            {
              v60 = v74;
              if ( v74 )
              {
                if ( v74 == 1 )
                  *v50 = src[0];
                else
                  memcpy(v50, src, v74);
                v60 = v74;
                v50 = (_BYTE *)*a7;
              }
              a7[1] = v60;
              v50[v60] = 0;
              v50 = v73;
            }
            else
            {
              v51 = v74;
              v12 = (__int64 *)src[0];
              if ( v50 == (_BYTE *)(a7 + 2) )
              {
                *a7 = v73;
                a7[1] = v51;
                a7[2] = v12;
              }
              else
              {
                v52 = a7[2];
                *a7 = v73;
                a7[1] = v51;
                a7[2] = v12;
                if ( v50 )
                {
                  v73 = v50;
                  src[0] = v52;
                  goto LABEL_74;
                }
              }
              v73 = src;
              v50 = src;
            }
LABEL_74:
            v74 = 0;
            *v50 = 0;
            v53 = v73;
            if ( v73 == src )
              goto LABEL_66;
            goto LABEL_75;
          }
          goto LABEL_17;
        }
        if ( v21 == 103 && n > 3 && v19[1] == 60 )
        {
          v34 = memchr(v19, 62, n);
          if ( v34 )
          {
            v35 = v34 - v19;
            v36 = v35;
            if ( v35 <= 1 )
            {
              v39 = 0;
              goto LABEL_78;
            }
            v12 = (__int64 *)n;
            if ( v35 != -1 )
            {
              v37 = n <= v35;
              v38 = n;
              if ( !v37 )
                v38 = v36;
              v39 = v38 - 2;
LABEL_78:
              v54 = v19 + 2;
              na = v36;
              if ( !(unsigned __int8)sub_C93C90(v54, v39, 10, v80) )
              {
                v12 = (__int64 *)v80[0];
                if ( *(_QWORD *)v80 == v80[0] )
                {
                  v55 = v69;
                  v28 = 0;
                  if ( na + 1 <= v69 )
                  {
                    v55 = na + 1;
                    v28 = v69 - (na + 1);
                  }
                  v69 = v28;
                  v68 += v55;
                  if ( (unsigned int)v84 > v80[0] )
                  {
                    v12 = (__int64 *)&v83[16 * v80[0]];
                    v49 = v12[1];
                    v48 = *v12;
                    if ( v49 > 0x3FFFFFFFFFFFFFFFLL - v71 )
                      goto LABEL_46;
                    goto LABEL_69;
                  }
                  if ( !a7 || a7[1] )
                    goto LABEL_23;
                  v78 = v39;
                  v76[0] = "invalid backreference string 'g<";
                  *(_QWORD *)v80 = v76;
                  v82 = 770;
                  v81 = ">'";
                  v77 = v54;
                  v79 = 1283;
                  sub_CA0F50(&v73, v80);
                  v56 = (_BYTE *)*a7;
                  v57 = (_BYTE *)*a7;
                  if ( v73 == src )
                  {
                    v61 = v74;
                    if ( v74 )
                    {
                      if ( v74 == 1 )
                        *v56 = src[0];
                      else
                        memcpy(v56, src, v74);
                      v61 = v74;
                      v56 = (_BYTE *)*a7;
                    }
                    a7[1] = v61;
                    v56[v61] = 0;
                    v57 = v73;
                  }
                  else
                  {
                    v58 = v74;
                    v12 = (__int64 *)src[0];
                    if ( v57 == (_BYTE *)(a7 + 2) )
                    {
                      *a7 = v73;
                      a7[1] = v58;
                      a7[2] = v12;
                    }
                    else
                    {
                      v59 = a7[2];
                      *a7 = v73;
                      a7[1] = v58;
                      a7[2] = v12;
                      if ( v57 )
                      {
                        v73 = v57;
                        src[0] = v59;
                        goto LABEL_89;
                      }
                    }
                    v73 = src;
                    v57 = src;
                  }
LABEL_89:
                  v74 = 0;
                  *v57 = 0;
                  v53 = v73;
                  if ( v73 == src )
                  {
LABEL_66:
                    v28 = v69;
                    goto LABEL_23;
                  }
LABEL_75:
                  j_j___libc_free_0(v53, src[0] + 1LL);
                  goto LABEL_66;
                }
              }
              v21 = *v68;
            }
          }
        }
LABEL_17:
        v22 = v71;
        v23 = (__int64)v70;
        v24 = 15;
        if ( v70 != &v72 )
          v24 = v72.m128i_i64[0];
        v25 = v71 + 1;
        if ( v71 + 1 > v24 )
        {
          nb = v71 + 1;
          sub_2240BB0(&v70, v71, 0, 0, 1);
          v23 = (__int64)v70;
          v25 = nb;
        }
        *(_BYTE *)(v23 + v22) = v21;
        v71 = v25;
        v70->m128i_i8[v22 + 1] = 0;
        v26 = v69;
        v27 = v68;
        if ( !v69 )
          goto LABEL_24;
        goto LABEL_22;
      }
      if ( v21 != 116 )
        goto LABEL_17;
      v30 = v71;
      v31 = (__int64)v70;
      v32 = 15;
      if ( v70 != &v72 )
        v32 = v72.m128i_i64[0];
      v33 = v71 + 1;
      if ( v71 + 1 > v32 )
      {
        sub_2240BB0(&v70, v71, 0, 0, 1);
        v31 = (__int64)v70;
      }
      *(_BYTE *)(v31 + v30) = 9;
    }
    v71 = v33;
    v70->m128i_i8[v30 + 1] = 0;
    v26 = v69;
    v27 = v68;
    if ( !v69 )
      goto LABEL_24;
LABEL_22:
    v28 = v26 - 1;
    v68 = v27 + 1;
    v69 = v28;
LABEL_23:
    if ( !v28 )
      goto LABEL_24;
  }
  v20 = v69;
  v17 = v68;
  if ( 0x3FFFFFFFFFFFFFFFLL - v71 < v69 )
    goto LABEL_46;
LABEL_50:
  sub_2241490(&v70, v17, v20, v14);
LABEL_51:
  if ( v69 != v20 && a7 && !a7[1] )
    sub_2241130(a7, 0, 0, "replacement string contained trailing backslash", 47);
LABEL_24:
  v9 = (char *)(*(_QWORD *)v83 + *((_QWORD *)v83 + 1));
  v29 = &a5[a6] - v9;
  if ( v29 > 0x3FFFFFFFFFFFFFFFLL - v71 )
    goto LABEL_46;
  sub_2241490(&v70, v9, v29, v12);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v70 == &v72 )
  {
    a1[1] = _mm_load_si128(&v72);
  }
  else
  {
    a1->m128i_i64[0] = (__int64)v70;
    a1[1].m128i_i64[0] = v72.m128i_i64[0];
  }
  a1->m128i_i64[1] = v71;
LABEL_3:
  if ( v83 != v85 )
    _libc_free(v83, v9);
  return a1;
}
