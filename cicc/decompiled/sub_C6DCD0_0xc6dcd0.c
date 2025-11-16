// Function: sub_C6DCD0
// Address: 0xc6dcd0
//
__int64 __fastcall sub_C6DCD0(__int64 *a1, unsigned __int16 *a2)
{
  unsigned __int8 *v4; // rdx
  unsigned __int8 *v5; // rsi
  __int64 v6; // rdi
  unsigned __int64 v7; // rcx
  unsigned __int8 *v8; // rax
  int v9; // r12d
  const char *v10; // rsi
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // r8
  int v16; // ecx
  char v17; // r10
  unsigned __int8 *v18; // rsi
  __int64 v19; // rax
  char *v20; // rdx
  size_t v21; // rsi
  __int64 v22; // rax
  char *v23; // rcx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  size_t v26; // rcx
  unsigned int v27; // r14d
  double v28; // xmm0_8
  __m128i **v29; // rsi
  char *v30; // rdi
  unsigned int v31; // eax
  __int16 **v32; // r12
  unsigned __int64 v33; // rdx
  unsigned __int8 *v34; // rcx
  unsigned __int8 *v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rbx
  __int16 *v38; // rsi
  unsigned __int8 *v39; // rdx
  unsigned __int8 *v40; // rsi
  unsigned __int64 v41; // rcx
  unsigned __int8 *v42; // rax
  _BYTE *v43; // rax
  _BYTE *v44; // rdx
  _BYTE *v45; // rax
  _BYTE *v46; // rdx
  _BYTE *v47; // rax
  _BYTE *v48; // rdx
  unsigned __int16 *v49; // rax
  unsigned __int8 *v50; // rcx
  __int64 v51; // rsi
  unsigned __int64 v52; // rdx
  unsigned __int8 *v53; // rax
  __int64 v54; // r15
  unsigned __int8 *v55; // rdx
  unsigned __int8 *v56; // rdi
  unsigned __int64 v57; // rsi
  unsigned __int8 *v58; // rax
  unsigned __int64 v59; // rdx
  size_t v60; // rax
  __int64 v61; // rax
  unsigned int v62; // eax
  char *v63; // r10
  unsigned __int8 *v64; // rdx
  unsigned __int8 *v65; // rdi
  unsigned __int64 v66; // rsi
  unsigned __int8 *v67; // rax
  unsigned __int8 v68; // dl
  __m128i *v69; // rdi
  __int64 v70; // rsi
  size_t v71; // rax
  unsigned __int64 v72; // rdx
  char *v73; // rdi
  __m128i *v74; // rax
  __int64 v75; // rdi
  size_t v76; // rdx
  char *v77; // [rsp+0h] [rbp-F0h]
  int *v78; // [rsp+10h] [rbp-E0h]
  __int64 v79; // [rsp+10h] [rbp-E0h]
  void *srca; // [rsp+18h] [rbp-D8h]
  char src; // [rsp+18h] [rbp-D8h]
  __m128i *v82; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v83; // [rsp+28h] [rbp-C8h]
  __m128i v84; // [rsp+30h] [rbp-C0h] BYREF
  char *endptr; // [rsp+40h] [rbp-B0h] BYREF
  size_t v86; // [rsp+48h] [rbp-A8h]
  __m128i v87; // [rsp+50h] [rbp-A0h] BYREF
  __m128i *v88; // [rsp+60h] [rbp-90h] BYREF
  size_t n; // [rsp+68h] [rbp-88h]
  __m128i v90; // [rsp+70h] [rbp-80h] BYREF
  char *nptr; // [rsp+90h] [rbp-60h] BYREF
  __int64 v92; // [rsp+98h] [rbp-58h]
  _OWORD v93[5]; // [rsp+A0h] [rbp-50h] BYREF

  v4 = (unsigned __int8 *)a1[3];
  v5 = (unsigned __int8 *)a1[4];
  if ( v5 != v4 )
  {
    v6 = 0x100002600LL;
    while ( 1 )
    {
      v7 = *v4;
      v8 = v4 + 1;
      if ( (unsigned __int8)v7 > 0x20u || !_bittest64(&v6, v7) )
        break;
      a1[3] = (__int64)v8;
      if ( v5 == v8 )
        goto LABEL_9;
      ++v4;
    }
    a1[3] = (__int64)v8;
    v9 = *v4;
    if ( (_BYTE)v9 != 34 )
    {
      switch ( (char)v9 )
      {
        case '[':
          v92 = 0;
          LOWORD(nptr) = 8;
          v93[0] = 0u;
          sub_C6BC50(a2);
          sub_C6A4F0((__int64)a2, (unsigned __int16 *)&nptr);
          v32 = (__int16 **)(a2 + 4);
          sub_C6BC50((unsigned __int16 *)&nptr);
          v34 = (unsigned __int8 *)a1[4];
          if ( *a2 != 8 )
            v32 = 0;
          v35 = (unsigned __int8 *)a1[3];
          if ( v34 == v35 )
            goto LABEL_38;
          v36 = 0x100002600LL;
          while ( 1 )
          {
            v33 = *v35;
            if ( (unsigned __int8)v33 > 0x20u )
              break;
            if ( _bittest64(&v36, v33) )
            {
              a1[3] = (__int64)++v35;
              if ( v35 != v34 )
                continue;
            }
            goto LABEL_38;
          }
          if ( (_BYTE)v33 == 93 )
            goto LABEL_141;
LABEL_38:
          v37 = 0x100002600LL;
          while ( 1 )
          {
            v38 = v32[1];
            if ( v38 == v32[2] )
            {
              sub_C6D0A0(v32, v38, v33);
              v38 = v32[1] - 20;
            }
            else
            {
              if ( v38 )
              {
                *v38 = 0;
                v38 = v32[1];
              }
              v32[1] = v38 + 20;
            }
            v27 = sub_C6DCD0(a1, v38);
            if ( !(_BYTE)v27 )
              return v27;
            v39 = (unsigned __int8 *)a1[3];
            v40 = (unsigned __int8 *)a1[4];
            if ( v40 == v39 )
              goto LABEL_48;
            while ( 1 )
            {
              v41 = *v39;
              v42 = v39 + 1;
              if ( (unsigned __int8)v41 > 0x20u || !_bittest64(&v37, v41) )
                break;
              a1[3] = (__int64)v42;
              if ( v40 == v42 )
                goto LABEL_48;
              ++v39;
            }
            a1[3] = (__int64)v42;
            v33 = *v39;
            if ( (_BYTE)v33 != 44 )
            {
              if ( (_BYTE)v33 == 93 )
                return v27;
LABEL_48:
              v10 = "Expected , or ] after array element";
              return sub_C68D40(a1, (__int64)v10);
            }
            for ( ; v42 != v40; a1[3] = (__int64)++v42 )
            {
              v33 = *v42;
              if ( (unsigned __int8)v33 > 0x20u )
                break;
              if ( !_bittest64(&v37, v33) )
                break;
            }
          }
        case 'f':
          LOBYTE(v92) = 0;
          LOWORD(nptr) = 1;
          sub_C6BC50(a2);
          sub_C6A4F0((__int64)a2, (unsigned __int16 *)&nptr);
          sub_C6BC50((unsigned __int16 *)&nptr);
          v43 = (_BYTE *)a1[3];
          v44 = (_BYTE *)a1[4];
          if ( v43 != v44 )
          {
            a1[3] = (__int64)(v43 + 1);
            if ( *v43 == 97 && v44 != v43 + 1 )
            {
              a1[3] = (__int64)(v43 + 2);
              if ( v43[1] == 108 && v44 != v43 + 2 )
              {
                a1[3] = (__int64)(v43 + 3);
                if ( v43[2] == 115 && v44 != v43 + 3 )
                {
                  a1[3] = (__int64)(v43 + 4);
                  if ( v43[3] == 101 )
                    return 1;
                }
              }
            }
          }
          v10 = "Invalid JSON value (false?)";
          return sub_C68D40(a1, (__int64)v10);
        case 'n':
          LOWORD(nptr) = 0;
          sub_C6BC50(a2);
          sub_C6A4F0((__int64)a2, (unsigned __int16 *)&nptr);
          sub_C6BC50((unsigned __int16 *)&nptr);
          v45 = (_BYTE *)a1[3];
          v46 = (_BYTE *)a1[4];
          if ( v46 != v45 )
          {
            a1[3] = (__int64)(v45 + 1);
            if ( *v45 == 117 && v46 != v45 + 1 )
            {
              a1[3] = (__int64)(v45 + 2);
              if ( v45[1] == 108 && v46 != v45 + 2 )
              {
                a1[3] = (__int64)(v45 + 3);
                if ( v45[2] == 108 )
                  return 1;
              }
            }
          }
          v10 = "Invalid JSON value (null?)";
          return sub_C68D40(a1, (__int64)v10);
        case 't':
          LOBYTE(v92) = 1;
          LOWORD(nptr) = 1;
          sub_C6BC50(a2);
          sub_C6A4F0((__int64)a2, (unsigned __int16 *)&nptr);
          sub_C6BC50((unsigned __int16 *)&nptr);
          v47 = (_BYTE *)a1[3];
          v48 = (_BYTE *)a1[4];
          if ( v47 != v48 )
          {
            a1[3] = (__int64)(v47 + 1);
            if ( *v47 == 114 && v48 != v47 + 1 )
            {
              a1[3] = (__int64)(v47 + 2);
              if ( v47[1] == 117 && v48 != v47 + 2 )
              {
                a1[3] = (__int64)(v47 + 3);
                if ( v47[2] == 101 )
                  return 1;
              }
            }
          }
          v10 = "Invalid JSON value (true?)";
          return sub_C68D40(a1, (__int64)v10);
        case '{':
          nptr = (char *)7;
          v92 = 1;
          memset(v93, 0, 24);
          v88 = (__m128i *)1;
          n = 0;
          v90.m128i_i64[0] = 0;
          v90.m128i_i32[2] = 0;
          sub_C6BC50(a2);
          sub_C6A4F0((__int64)a2, (unsigned __int16 *)&nptr);
          sub_C6BC50((unsigned __int16 *)&nptr);
          sub_C6B900((__int64)&v88);
          sub_C7D6A0(n, (unsigned __int64)v90.m128i_u32[2] << 6, 8);
          v49 = 0;
          if ( *a2 == 7 )
            v49 = a2 + 4;
          v50 = (unsigned __int8 *)a1[4];
          v79 = (__int64)v49;
          v35 = (unsigned __int8 *)a1[3];
          if ( v50 == v35 )
            goto LABEL_99;
          v51 = 0x100002600LL;
          while ( 1 )
          {
            v52 = *v35;
            if ( (unsigned __int8)v52 > 0x20u )
              break;
            if ( !_bittest64(&v51, v52) )
              goto LABEL_67;
            a1[3] = (__int64)++v35;
            if ( v35 == v50 )
              goto LABEL_99;
          }
          if ( (_BYTE)v52 != 125 )
          {
LABEL_67:
            v53 = (unsigned __int8 *)a1[3];
            if ( v53 != v50 )
            {
              v54 = 0x100002600LL;
              while ( 1 )
              {
                a1[3] = (__int64)(v53 + 1);
                if ( *v53 != 34 )
                  goto LABEL_99;
                v90.m128i_i8[0] = 0;
                v88 = &v90;
                n = 0;
                if ( !(unsigned __int8)sub_C69E50(a1, &v88) )
                  break;
                v55 = (unsigned __int8 *)a1[3];
                v56 = (unsigned __int8 *)a1[4];
                if ( v55 == v56 )
                  goto LABEL_110;
                while ( 1 )
                {
                  v57 = *v55;
                  v58 = v55 + 1;
                  if ( (unsigned __int8)v57 > 0x20u || !_bittest64(&v54, v57) )
                    break;
                  a1[3] = (__int64)v58;
                  if ( v56 == v58 )
                    goto LABEL_110;
                  ++v55;
                }
                a1[3] = (__int64)v58;
                if ( *v55 != 58 )
                {
LABEL_110:
                  v27 = sub_C68D40(a1, (__int64)"Expected : after object key");
                  goto LABEL_111;
                }
                for ( ; v56 != v58; a1[3] = (__int64)++v58 )
                {
                  v59 = *v58;
                  if ( (unsigned __int8)v59 > 0x20u )
                    break;
                  if ( !_bittest64(&v54, v59) )
                    break;
                }
                nptr = (char *)v93;
                if ( v88 == &v90 )
                {
                  v93[0] = _mm_load_si128(&v90);
                }
                else
                {
                  nptr = (char *)v88;
                  *(_QWORD *)&v93[0] = v90.m128i_i64[0];
                }
                v60 = n;
                v90.m128i_i8[0] = 0;
                n = 0;
                v92 = v60;
                v88 = &v90;
                sub_C6B270((__int64 **)&endptr, (__int64)&nptr);
                v61 = sub_C6DC20(v79, (__int64)&endptr);
                v62 = sub_C6DCD0(a1, v61);
                v63 = endptr;
                v27 = v62;
                if ( endptr )
                {
                  if ( *(char **)endptr != endptr + 16 )
                  {
                    v77 = endptr;
                    j_j___libc_free_0(*(_QWORD *)endptr, *((_QWORD *)endptr + 2) + 1LL);
                    v63 = v77;
                  }
                  j_j___libc_free_0(v63, 32);
                }
                if ( nptr != (char *)v93 )
                  j_j___libc_free_0(nptr, *(_QWORD *)&v93[0] + 1LL);
                if ( !(_BYTE)v27 )
                  break;
                v64 = (unsigned __int8 *)a1[3];
                v65 = (unsigned __int8 *)a1[4];
                if ( v65 == v64 )
                  goto LABEL_89;
                while ( 1 )
                {
                  v66 = *v64;
                  v67 = v64 + 1;
                  if ( (unsigned __int8)v66 > 0x20u || !_bittest64(&v54, v66) )
                    break;
                  a1[3] = (__int64)v67;
                  if ( v67 == v65 )
                    goto LABEL_89;
                  ++v64;
                }
                a1[3] = (__int64)v67;
                v68 = *v64;
                if ( v68 != 44 )
                {
                  if ( v68 != 125 )
LABEL_89:
                    v27 = sub_C68D40(a1, (__int64)"Expected , or } after object property");
LABEL_111:
                  if ( v88 != &v90 )
                    j_j___libc_free_0(v88, v90.m128i_i64[0] + 1);
                  return v27;
                }
                for ( ; v67 != v65; a1[3] = (__int64)++v67 )
                {
                  v72 = *v67;
                  if ( (unsigned __int8)v72 > 0x20u )
                    break;
                  if ( !_bittest64(&v54, v72) )
                    break;
                }
                if ( v88 != &v90 )
                {
                  j_j___libc_free_0(v88, v90.m128i_i64[0] + 1);
                  v65 = (unsigned __int8 *)a1[4];
                }
                v53 = (unsigned __int8 *)a1[3];
                if ( v53 == v65 )
                  goto LABEL_99;
              }
              v27 = 0;
              goto LABEL_111;
            }
LABEL_99:
            v10 = "Expected object key";
            return sub_C68D40(a1, (__int64)v10);
          }
LABEL_141:
          a1[3] = (__int64)(v35 + 1);
          return 1;
        default:
          if ( (unsigned __int8)(v9 - 43) > 0x3Au
            || (v12 = 0x400000004007FEDLL, !_bittest64(&v12, (unsigned int)(v9 - 43))) )
          {
            v10 = "Invalid JSON value";
            return sub_C68D40(a1, (__int64)v10);
          }
          *(_QWORD *)&v93[0] = 24;
          v13 = 24;
          v14 = 1;
          v15 = 0x400000004007FE1LL;
          nptr = (char *)v93 + 8;
          BYTE8(v93[0]) = v9;
          v92 = 1;
          if ( v8 != v5 )
          {
            do
            {
              v16 = *v8;
              if ( (unsigned __int8)(v16 - 43) > 0x3Au
                || !_bittest64(&v15, (unsigned int)(v16 - 43)) && (unsigned __int8)(v16 - 45) > 1u )
              {
                break;
              }
              a1[3] = (__int64)(v8 + 1);
              v17 = *v8;
              if ( v14 + 1 > v13 )
              {
                src = *v8;
                sub_C8D290(&nptr, (char *)v93 + 8, v14 + 1, 1);
                v14 = v92;
                v17 = src;
                v15 = 0x400000004007FE1LL;
              }
              nptr[v14] = v17;
              v18 = (unsigned __int8 *)a1[4];
              v13 = *(_QWORD *)&v93[0];
              v14 = v92 + 1;
              v8 = (unsigned __int8 *)a1[3];
              ++v92;
            }
            while ( v8 != v18 );
          }
          srca = (void *)v13;
          v78 = __errno_location();
          *v78 = 0;
          if ( v14 + 1 > (unsigned __int64)srca )
          {
            sub_C8D290(&nptr, (char *)v93 + 8, v14 + 1, 1);
            v14 = v92;
          }
          nptr[v14] = 0;
          v19 = strtoll(nptr, &endptr, 10);
          v20 = nptr;
          v21 = v19;
          v22 = v92;
          v23 = &nptr[v92];
          if ( &nptr[v92] == endptr && *v78 != 34 )
          {
            n = v21;
            LOWORD(v88) = 3;
LABEL_129:
            v27 = 1;
            sub_C6BC50(a2);
            v29 = &v88;
            sub_C6A4F0((__int64)a2, (unsigned __int16 *)&v88);
            sub_C6BC50((unsigned __int16 *)&v88);
            v30 = nptr;
            goto LABEL_30;
          }
          if ( (_BYTE)v9 != 45 )
          {
            v24 = v92 + 1;
            *v78 = 0;
            if ( v24 > *(_QWORD *)&v93[0] )
            {
              sub_C8D290(&nptr, (char *)v93 + 8, v24, 1);
              v23 = &nptr[v92];
            }
            *v23 = 0;
            v25 = strtoull(nptr, &endptr, 10);
            v20 = nptr;
            v26 = v25;
            v22 = v92;
            if ( endptr == &nptr[v92] && *v78 != 34 )
            {
              n = v26;
              LOWORD(v88) = 4;
              goto LABEL_129;
            }
          }
          if ( (unsigned __int64)(v22 + 1) > *(_QWORD *)&v93[0] )
          {
            sub_C8D290(&nptr, (char *)v93 + 8, v22 + 1, 1);
            v22 = v92;
            v20 = nptr;
          }
          v20[v22] = 0;
          v27 = 1;
          v28 = strtod(nptr, &endptr);
          LOWORD(v88) = 2;
          n = *(_QWORD *)&v28;
          sub_C6BC50(a2);
          v29 = &v88;
          sub_C6A4F0((__int64)a2, (unsigned __int16 *)&v88);
          sub_C6BC50((unsigned __int16 *)&v88);
          v30 = nptr;
          if ( endptr != &nptr[v92] )
          {
            v29 = (__m128i **)"Invalid JSON value (number?)";
            v31 = sub_C68D40(a1, (__int64)"Invalid JSON value (number?)");
            v30 = nptr;
            v27 = v31;
          }
LABEL_30:
          if ( v30 != (char *)v93 + 8 )
            _libc_free(v30, v29);
          break;
      }
      return v27;
    }
    v83 = 0;
    v82 = &v84;
    v84.m128i_i8[0] = 0;
    v27 = sub_C69E50(a1, &v82);
    if ( !(_BYTE)v27 )
    {
LABEL_107:
      if ( v82 != &v84 )
        j_j___libc_free_0(v82, v84.m128i_i64[0] + 1);
      return v27;
    }
    v69 = v82;
    endptr = (char *)&v87;
    if ( v82 == &v84 )
    {
      v69 = &v87;
      v87 = _mm_load_si128(&v84);
    }
    else
    {
      endptr = (char *)v82;
      v87.m128i_i64[0] = v84.m128i_i64[0];
    }
    v70 = v83;
    v82 = &v84;
    v83 = 0;
    v86 = v70;
    v84.m128i_i8[0] = 0;
    LOWORD(nptr) = 6;
    if ( (unsigned __int8)sub_C6A630(v69->m128i_i8, v70, 0) )
    {
LABEL_137:
      v92 = (__int64)v93 + 8;
      if ( endptr == (char *)&v87 )
      {
        *(__m128i *)((char *)v93 + 8) = _mm_load_si128(&v87);
      }
      else
      {
        v92 = (__int64)endptr;
        *((_QWORD *)&v93[0] + 1) = v87.m128i_i64[0];
      }
      v71 = v86;
      endptr = (char *)&v87;
      v86 = 0;
      *(_QWORD *)&v93[0] = v71;
      v87.m128i_i8[0] = 0;
      sub_C6BC50(a2);
      sub_C6A4F0((__int64)a2, (unsigned __int16 *)&nptr);
      sub_C6BC50((unsigned __int16 *)&nptr);
      if ( endptr != (char *)&v87 )
        j_j___libc_free_0(endptr, v87.m128i_i64[0] + 1);
      goto LABEL_107;
    }
    sub_C6B0E0((__int64 *)&v88, (__int64)endptr, v86);
    v73 = endptr;
    v74 = (__m128i *)endptr;
    if ( v88 == &v90 )
    {
      v76 = n;
      if ( n )
      {
        if ( n == 1 )
          *endptr = v90.m128i_i8[0];
        else
          memcpy(endptr, &v90, n);
        v73 = endptr;
        v76 = n;
      }
      v86 = v76;
      v73[v76] = 0;
      v74 = v88;
      goto LABEL_166;
    }
    if ( endptr == (char *)&v87 )
    {
      endptr = (char *)v88;
      v86 = n;
      v87.m128i_i64[0] = v90.m128i_i64[0];
    }
    else
    {
      v75 = v87.m128i_i64[0];
      endptr = (char *)v88;
      v86 = n;
      v87.m128i_i64[0] = v90.m128i_i64[0];
      if ( v74 )
      {
        v88 = v74;
        v90.m128i_i64[0] = v75;
        goto LABEL_166;
      }
    }
    v88 = &v90;
    v74 = &v90;
LABEL_166:
    n = 0;
    v74->m128i_i8[0] = 0;
    if ( v88 != &v90 )
      j_j___libc_free_0(v88, v90.m128i_i64[0] + 1);
    goto LABEL_137;
  }
LABEL_9:
  v10 = "Unexpected EOF";
  return sub_C68D40(a1, (__int64)v10);
}
