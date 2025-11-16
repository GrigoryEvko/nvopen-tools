// Function: sub_BEA6A0
// Address: 0xbea6a0
//
void __fastcall sub_BEA6A0(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 *v5; // rbx
  __int64 v6; // r12
  int v7; // eax
  __m128i *v8; // rax
  __int64 v9; // rcx
  __m128i *v10; // rax
  int v11; // ebx
  int v12; // ebx
  int v13; // ebx
  char v14; // r8
  int v15; // eax
  int v16; // ebx
  int v17; // r12d
  __int64 *v18; // rbx
  unsigned int v19; // eax
  __m128i *v20; // rax
  __int64 v21; // rcx
  __m128i *v22; // rax
  __int64 v23; // rsi
  const char *v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 *v27; // rdi
  const char *v28; // rax
  unsigned __int16 v29; // ax
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned int v34; // r8d
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned int v39; // r8d
  __int64 v40; // r12
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  unsigned int v46; // r8d
  __int64 v47; // r12
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  unsigned int v53; // r8d
  __int64 v54; // r12
  __int64 v55; // rsi
  __int64 v56; // rdx
  const char *v57; // rax
  __int64 v59; // [rsp+8h] [rbp-168h]
  _BYTE *v60; // [rsp+10h] [rbp-160h] BYREF
  __int64 v61; // [rsp+18h] [rbp-158h] BYREF
  __int64 v62; // [rsp+20h] [rbp-150h] BYREF
  __int64 v63; // [rsp+28h] [rbp-148h] BYREF
  _QWORD v64[2]; // [rsp+30h] [rbp-140h] BYREF
  __int64 v65; // [rsp+40h] [rbp-130h] BYREF
  _BYTE v66[32]; // [rsp+50h] [rbp-120h] BYREF
  __m128i *v67; // [rsp+70h] [rbp-100h] BYREF
  __int64 v68; // [rsp+78h] [rbp-F8h]
  __m128i v69; // [rsp+80h] [rbp-F0h] BYREF
  __m128i *v70; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v71; // [rsp+98h] [rbp-D8h]
  __m128i v72; // [rsp+A0h] [rbp-D0h] BYREF
  char v73; // [rsp+B0h] [rbp-C0h]
  char v74; // [rsp+B1h] [rbp-BFh]
  _QWORD v75[2]; // [rsp+C0h] [rbp-B0h] BYREF
  __m128i v76; // [rsp+D0h] [rbp-A0h] BYREF
  _QWORD *v77; // [rsp+E0h] [rbp-90h]
  const char *v78; // [rsp+100h] [rbp-70h] BYREF
  _WORD *v79; // [rsp+108h] [rbp-68h]
  __int64 v80; // [rsp+110h] [rbp-60h]
  int v81; // [rsp+118h] [rbp-58h]
  char v82; // [rsp+11Ch] [rbp-54h]
  _WORD v83[40]; // [rsp+120h] [rbp-50h] BYREF

  v61 = a2;
  v60 = a4;
  if ( a2 )
  {
    sub_BE7B90(a1, a2, a4);
    v5 = (__int64 *)sub_A73280(&v61);
    v6 = sub_A73290(&v61);
    if ( v5 == (__int64 *)v6 )
    {
LABEL_19:
      if ( (unsigned __int8)sub_A73170(&v61, 14) && (unsigned int)sub_A73160((__int64)&v61) != 1 )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attribute 'immarg' is incompatible with other attributes";
LABEL_53:
        v78 = v24;
        LOBYTE(v83[0]) = 3;
        sub_BE7760(a1, (__int64)&v78, &v60);
        return;
      }
      v11 = (unsigned __int8)sub_A73170(&v61, 81);
      v12 = (unsigned __int8)sub_A73170(&v61, 83) + v11;
      v13 = (unsigned __int8)sub_A73170(&v61, 84) + v12;
      v14 = sub_A73170(&v61, 85);
      v15 = 1;
      if ( !v14 )
        v15 = (unsigned __int8)sub_A73170(&v61, 15);
      v16 = v15 + v13;
      v17 = (unsigned __int8)sub_A73170(&v61, 21);
      if ( v16 + v17 + (unsigned int)(unsigned __int8)sub_A73170(&v61, 80) > 1 )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes 'byval', 'inalloca', 'preallocated', 'inreg', 'nest', 'byref', and 'sret' are incompatible!";
        goto LABEL_53;
      }
      if ( (unsigned __int8)sub_A73170(&v61, 83) && (unsigned __int8)sub_A73170(&v61, 51) )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes 'inalloca and readonly' are incompatible!";
        goto LABEL_53;
      }
      if ( (unsigned __int8)sub_A73170(&v61, 85) && (unsigned __int8)sub_A73170(&v61, 52) )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes 'sret and returned' are incompatible!";
        goto LABEL_53;
      }
      if ( (unsigned __int8)sub_A73170(&v61, 79) && (unsigned __int8)sub_A73170(&v61, 54) )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes 'zeroext and signext' are incompatible!";
        goto LABEL_53;
      }
      if ( (unsigned __int8)sub_A73170(&v61, 50) && (unsigned __int8)sub_A73170(&v61, 51) )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes 'readnone and readonly' are incompatible!";
        goto LABEL_53;
      }
      if ( (unsigned __int8)sub_A73170(&v61, 50) && (unsigned __int8)sub_A73170(&v61, 78) )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes 'readnone and writeonly' are incompatible!";
        goto LABEL_53;
      }
      if ( (unsigned __int8)sub_A73170(&v61, 51) && (unsigned __int8)sub_A73170(&v61, 78) )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes 'readonly and writeonly' are incompatible!";
        goto LABEL_53;
      }
      if ( (unsigned __int8)sub_A73170(&v61, 31) && (unsigned __int8)sub_A73170(&v61, 3) )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes 'noinline and alwaysinline' are incompatible!";
        goto LABEL_53;
      }
      if ( (unsigned __int8)sub_A73170(&v61, 77) && (unsigned __int8)sub_A73170(&v61, 50) )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes writable and readnone are incompatible!";
        goto LABEL_53;
      }
      if ( (unsigned __int8)sub_A73170(&v61, 77) && (unsigned __int8)sub_A73170(&v61, 51) )
      {
        HIBYTE(v83[0]) = 1;
        v24 = "Attributes writable and readonly are incompatible!";
        goto LABEL_53;
      }
      sub_A751C0((__int64)v75, a3, v61, 3);
      v18 = (__int64 *)sub_A73280(&v61);
      v59 = sub_A73290(&v61);
      if ( v18 == (__int64 *)v59 )
      {
LABEL_57:
        if ( *(_BYTE *)(a3 + 8) != 14 )
        {
LABEL_58:
          if ( !(unsigned __int8)sub_A73170(&v61, 98) )
            goto LABEL_59;
          v78 = (const char *)sub_A734C0(&v61, 98);
          v27 = (__int64 *)sub_A72F20((__int64 *)&v78);
          if ( !v26 )
          {
            HIBYTE(v83[0]) = 1;
            v28 = "Attribute 'initializes' does not support empty list";
            goto LABEL_70;
          }
          if ( (unsigned __int8)sub_ABEE90(v27, v26) )
          {
LABEL_59:
            if ( !(unsigned __int8)sub_A73170(&v61, 93) )
              goto LABEL_62;
            v78 = (const char *)sub_A734C0(&v61, 93);
            v25 = sub_A71B80((__int64 *)&v78);
            if ( v25 )
            {
              if ( (v25 & 0xFFFFFC00) == 0 )
              {
LABEL_62:
                v23 = 97;
                if ( !(unsigned __int8)sub_A73170(&v61, 97) )
                  goto LABEL_66;
                v78 = (const char *)sub_A734C0(&v61, 97);
                v23 = *(unsigned int *)(sub_A72A90((__int64 *)&v78) + 8);
                if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
                  a3 = **(_QWORD **)(a3 + 16);
                if ( sub_BCAC40(a3, v23) )
                  goto LABEL_66;
                HIBYTE(v83[0]) = 1;
                v28 = "Range bit width must match type bit width!";
                goto LABEL_70;
              }
              HIBYTE(v83[0]) = 1;
              v28 = "Invalid value for 'nofpclass' test mask";
            }
            else
            {
              HIBYTE(v83[0]) = 1;
              v28 = "Attribute 'nofpclass' must have at least one test bit set";
            }
          }
          else
          {
            HIBYTE(v83[0]) = 1;
            v28 = "Attribute 'initializes' does not support unordered ranges";
          }
LABEL_70:
          v23 = (__int64)&v78;
          v78 = v28;
          LOBYTE(v83[0]) = 3;
          sub_BE7760(a1, (__int64)&v78, &v60);
LABEL_66:
          sub_BD9B10(v77, v23);
          return;
        }
        if ( (unsigned __int8)sub_A73170(&v61, 86) )
        {
          v29 = sub_A73630(&v61);
          if ( HIBYTE(v29) )
          {
            if ( (unsigned __int64)(1LL << v29) > 0x100000000LL )
            {
              v23 = (__int64)&v78;
              v78 = "huge alignment values are unsupported";
              v83[0] = 259;
              sub_BE7760(a1, (__int64)&v78, &v60);
              goto LABEL_66;
            }
          }
        }
        if ( (unsigned __int8)sub_A73170(&v61, 81) )
        {
          v30 = sub_A73700(&v61);
          v82 = 1;
          v31 = v30;
          v78 = 0;
          v79 = v83;
          v80 = 4;
          v81 = 0;
          if ( !(unsigned __int8)sub_9C6430(v30, (__int64)&v78, v32, v33, v34) )
          {
            v74 = 1;
            v57 = "Attribute 'byval' does not support unsized types!";
            goto LABEL_112;
          }
          if ( (unsigned __int8)sub_BCF0D0(v31) )
          {
            v74 = 1;
            v57 = "'byval' argument has illegal target extension type";
            goto LABEL_112;
          }
          v70 = (__m128i *)sub_BDB740(a1[17], v31);
          v71 = v35;
          if ( (unsigned __int64)v70 > 0xFFFFFFFF )
          {
            v74 = 1;
            v57 = "huge 'byval' arguments are unsupported";
            goto LABEL_112;
          }
          if ( !v82 )
            _libc_free(v79, v31);
        }
        if ( (unsigned __int8)sub_A73170(&v61, 80) )
        {
          v82 = 1;
          v78 = 0;
          v79 = v83;
          v80 = 4;
          v81 = 0;
          v36 = sub_A736E0(&v61);
          if ( !(unsigned __int8)sub_9C6430(v36, (__int64)&v78, v37, v38, v39) )
          {
            v74 = 1;
            v57 = "Attribute 'byref' does not support unsized types!";
            goto LABEL_112;
          }
          v40 = a1[17];
          v41 = sub_A736E0(&v61);
          v70 = (__m128i *)sub_BDB740(v40, v41);
          v71 = v42;
          if ( (unsigned __int64)v70 > 0xFFFFFFFF )
          {
            v74 = 1;
            v57 = "huge 'byref' arguments are unsupported";
            goto LABEL_112;
          }
          if ( !v82 )
            _libc_free(v79, v41);
        }
        if ( (unsigned __int8)sub_A73170(&v61, 83) )
        {
          v82 = 1;
          v78 = 0;
          v79 = v83;
          v80 = 4;
          v81 = 0;
          v43 = sub_A73760(&v61);
          if ( !(unsigned __int8)sub_9C6430(v43, (__int64)&v78, v44, v45, v46) )
          {
            v74 = 1;
            v57 = "Attribute 'inalloca' does not support unsized types!";
            goto LABEL_112;
          }
          v47 = a1[17];
          v48 = sub_A73760(&v61);
          v70 = (__m128i *)sub_BDB740(v47, v48);
          v71 = v49;
          if ( (unsigned __int64)v70 > 0xFFFFFFFF )
          {
            v74 = 1;
            v57 = "huge 'inalloca' arguments are unsupported";
            goto LABEL_112;
          }
          if ( !v82 )
            _libc_free(v79, v48);
        }
        if ( !(unsigned __int8)sub_A73170(&v61, 84) )
          goto LABEL_58;
        v82 = 1;
        v78 = 0;
        v79 = v83;
        v80 = 4;
        v81 = 0;
        v50 = sub_A73740(&v61);
        if ( (unsigned __int8)sub_9C6430(v50, (__int64)&v78, v51, v52, v53) )
        {
          v54 = a1[17];
          v55 = sub_A73740(&v61);
          v70 = (__m128i *)sub_BDB740(v54, v55);
          v71 = v56;
          if ( (unsigned __int64)v70 <= 0xFFFFFFFF )
          {
            if ( !v82 )
              _libc_free(v79, v55);
            goto LABEL_58;
          }
          v74 = 1;
          v57 = "huge 'preallocated' arguments are unsupported";
        }
        else
        {
          v74 = 1;
          v57 = "Attribute 'preallocated' does not support unsized types!";
        }
LABEL_112:
        v23 = (__int64)&v70;
        v70 = (__m128i *)v57;
        v73 = 3;
        sub_BE7760(a1, (__int64)&v70, &v60);
        if ( !v82 )
          _libc_free(v79, &v70);
        goto LABEL_66;
      }
      while ( 1 )
      {
        v63 = *v18;
        if ( !sub_A71840((__int64)&v63) )
        {
          v19 = sub_A71AE0(&v63);
          if ( (v75[(unsigned __int64)v19 >> 6] & (1LL << v19)) != 0 )
            break;
        }
        if ( (__int64 *)v59 == ++v18 )
          goto LABEL_57;
      }
      sub_A759D0((__int64)v66, &v63, 0);
      v20 = (__m128i *)sub_2241130(v66, 0, 0, "Attribute '", 11);
      v67 = &v69;
      if ( (__m128i *)v20->m128i_i64[0] == &v20[1] )
      {
        v69 = _mm_loadu_si128(v20 + 1);
      }
      else
      {
        v67 = (__m128i *)v20->m128i_i64[0];
        v69.m128i_i64[0] = v20[1].m128i_i64[0];
      }
      v21 = v20->m128i_i64[1];
      v20[1].m128i_i8[0] = 0;
      v68 = v21;
      v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
      v20->m128i_i64[1] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v68) > 0x1E )
      {
        v22 = (__m128i *)sub_2241490(&v67, "' applied to incompatible type!", 31, v21);
        v70 = &v72;
        if ( (__m128i *)v22->m128i_i64[0] == &v22[1] )
        {
          v72 = _mm_loadu_si128(v22 + 1);
        }
        else
        {
          v70 = (__m128i *)v22->m128i_i64[0];
          v72.m128i_i64[0] = v22[1].m128i_i64[0];
        }
        v23 = (__int64)&v78;
        v71 = v22->m128i_i64[1];
        v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
        v22->m128i_i64[1] = 0;
        v22[1].m128i_i8[0] = 0;
        v83[0] = 260;
        v78 = (const char *)&v70;
        sub_BE7760(a1, (__int64)&v78, &v60);
        sub_2240A30(&v70);
        sub_2240A30(&v67);
        sub_2240A30(v66);
        goto LABEL_66;
      }
LABEL_122:
      sub_4262D8((__int64)"basic_string::append");
    }
    while ( 1 )
    {
      v62 = *v5;
      if ( !sub_A71840((__int64)&v62) )
      {
        v7 = sub_A71AE0(&v62);
        if ( !sub_A71A10(v7) )
          break;
      }
      if ( (__int64 *)v6 == ++v5 )
        goto LABEL_19;
    }
    sub_A759D0((__int64)v64, &v62, 0);
    v8 = (__m128i *)sub_2241130(v64, 0, 0, "Attribute '", 11);
    v70 = &v72;
    if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
    {
      v72 = _mm_loadu_si128(v8 + 1);
    }
    else
    {
      v70 = (__m128i *)v8->m128i_i64[0];
      v72.m128i_i64[0] = v8[1].m128i_i64[0];
    }
    v9 = v8->m128i_i64[1];
    v71 = v9;
    v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
    v8->m128i_i64[1] = 0;
    v8[1].m128i_i8[0] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v71) <= 0x1D )
      goto LABEL_122;
    v10 = (__m128i *)sub_2241490(&v70, "' does not apply to parameters", 30, v9);
    v75[0] = &v76;
    if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
    {
      v76 = _mm_loadu_si128(v10 + 1);
    }
    else
    {
      v75[0] = v10->m128i_i64[0];
      v76.m128i_i64[0] = v10[1].m128i_i64[0];
    }
    v75[1] = v10->m128i_i64[1];
    v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
    v10->m128i_i64[1] = 0;
    v10[1].m128i_i8[0] = 0;
    v83[0] = 260;
    v78 = (const char *)v75;
    sub_BE7760(a1, (__int64)&v78, &v60);
    if ( (__m128i *)v75[0] != &v76 )
      j_j___libc_free_0(v75[0], v76.m128i_i64[0] + 1);
    if ( v70 != &v72 )
      j_j___libc_free_0(v70, v72.m128i_i64[0] + 1);
    if ( (__int64 *)v64[0] != &v65 )
      j_j___libc_free_0(v64[0], v65 + 1);
  }
}
