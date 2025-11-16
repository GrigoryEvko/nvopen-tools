// Function: sub_C69E50
// Address: 0xc69e50
//
__int64 __fastcall sub_C69E50(__int64 *a1, _QWORD *a2)
{
  char *v4; // rcx
  char *v5; // rax
  char *v6; // rdx
  char v7; // r12
  _QWORD *v8; // r13
  __int64 v9; // r15
  _QWORD *v10; // rax
  unsigned __int64 v11; // r11
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rax
  const char *v14; // rsi
  unsigned __int8 v16; // r15
  char v17; // al
  unsigned __int16 v18; // r12
  __int16 *i; // rsi
  __int64 v20; // rcx
  __int16 v21; // di
  __int64 v22; // r12
  _QWORD *v23; // rax
  unsigned __int64 v24; // r15
  unsigned __int64 v25; // rdx
  _QWORD *v26; // rax
  __int64 v27; // r12
  _QWORD *v28; // rax
  unsigned __int64 v29; // r15
  unsigned __int64 v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // r12
  _QWORD *v33; // rax
  unsigned __int64 v34; // r15
  unsigned __int64 v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // r12
  _QWORD *v38; // rax
  unsigned __int64 v39; // r15
  unsigned __int64 v40; // rdx
  _QWORD *v41; // rax
  __int64 v42; // r12
  _QWORD *v43; // rax
  unsigned __int64 v44; // r15
  unsigned __int64 v45; // rdx
  _QWORD *v46; // rax
  __int64 v47; // rcx
  _BYTE *v48; // rax
  unsigned __int64 v49; // rsi
  _BYTE *v50; // rcx
  unsigned __int8 v51; // cl
  char *v52; // rax
  char v53; // di
  char v54; // di
  char v55; // di
  __int16 *v56; // rsi
  _BYTE *v57; // r8
  unsigned __int16 v58; // r15
  int v59; // eax
  __int16 v60; // ax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rax
  __int16 *v64; // [rsp+0h] [rbp-60h]
  _BYTE *v65; // [rsp+10h] [rbp-50h]
  __int16 *v66; // [rsp+18h] [rbp-48h]
  unsigned __int8 v67; // [rsp+18h] [rbp-48h]
  __int16 v68; // [rsp+26h] [rbp-3Ah] BYREF
  char v69; // [rsp+28h] [rbp-38h]
  __int16 v70; // [rsp+29h] [rbp-37h] BYREF
  char v71; // [rsp+2Bh] [rbp-35h]
  __int16 v72; // [rsp+2Ch] [rbp-34h] BYREF
  char v73; // [rsp+2Eh] [rbp-32h]
  char v74; // [rsp+2Fh] [rbp-31h]
  _BYTE v75[48]; // [rsp+30h] [rbp-30h] BYREF

  v4 = (char *)a1[3];
  v5 = (char *)a1[4];
  if ( v4 == v5 )
  {
    v6 = (char *)a1[4];
    v7 = 0;
    goto LABEL_3;
  }
  while ( 1 )
  {
    v6 = v4 + 1;
    a1[3] = (__int64)(v4 + 1);
    v7 = *v4;
    if ( *v4 == 34 )
      return 1;
LABEL_3:
    v8 = a2 + 2;
    if ( v6 == v5 )
      goto LABEL_12;
    if ( (v7 & 0xE0) == 0 )
    {
      v14 = "Control character in string";
      return sub_C68D40(a1, (__int64)v14);
    }
    if ( v7 != 92 )
      goto LABEL_6;
    a1[3] = (__int64)(v6 + 1);
    v7 = *v6;
    if ( *v6 > 117 )
      goto LABEL_22;
    if ( v7 > 91 )
      break;
    if ( v7 != 34 && v7 != 47 )
    {
LABEL_22:
      v14 = "Invalid escape sequence";
      return sub_C68D40(a1, (__int64)v14);
    }
LABEL_6:
    v9 = a2[1];
    v10 = (_QWORD *)*a2;
    v11 = v9 + 1;
    if ( (_QWORD *)*a2 == v8 )
      v12 = 15;
    else
      v12 = a2[2];
    if ( v11 > v12 )
    {
      sub_2240BB0(a2, a2[1], 0, 0, 1);
      v10 = (_QWORD *)*a2;
      v11 = v9 + 1;
    }
    *((_BYTE *)v10 + v9) = v7;
    v13 = (_QWORD *)*a2;
    a2[1] = v11;
    *((_BYTE *)v13 + v9 + 1) = 0;
LABEL_11:
    v4 = (char *)a1[3];
    v5 = (char *)a1[4];
    if ( v4 == v5 )
    {
LABEL_12:
      v14 = "Unterminated string";
      return sub_C68D40(a1, (__int64)v14);
    }
  }
  switch ( v7 )
  {
    case '\\':
      goto LABEL_6;
    case 'b':
      v42 = a2[1];
      v43 = (_QWORD *)*a2;
      v44 = v42 + 1;
      if ( (_QWORD *)*a2 == v8 )
        v45 = 15;
      else
        v45 = a2[2];
      if ( v44 > v45 )
      {
        sub_2240BB0(a2, a2[1], 0, 0, 1);
        v43 = (_QWORD *)*a2;
      }
      *((_BYTE *)v43 + v42) = 8;
      v46 = (_QWORD *)*a2;
      a2[1] = v44;
      *((_BYTE *)v46 + v42 + 1) = 0;
      goto LABEL_11;
    case 'f':
      v37 = a2[1];
      v38 = (_QWORD *)*a2;
      v39 = v37 + 1;
      if ( (_QWORD *)*a2 == v8 )
        v40 = 15;
      else
        v40 = a2[2];
      if ( v39 > v40 )
      {
        sub_2240BB0(a2, a2[1], 0, 0, 1);
        v38 = (_QWORD *)*a2;
      }
      *((_BYTE *)v38 + v37) = 12;
      v41 = (_QWORD *)*a2;
      a2[1] = v39;
      *((_BYTE *)v41 + v37 + 1) = 0;
      goto LABEL_11;
    case 'n':
      v32 = a2[1];
      v33 = (_QWORD *)*a2;
      v34 = v32 + 1;
      if ( (_QWORD *)*a2 == v8 )
        v35 = 15;
      else
        v35 = a2[2];
      if ( v34 > v35 )
      {
        sub_2240BB0(a2, a2[1], 0, 0, 1);
        v33 = (_QWORD *)*a2;
      }
      *((_BYTE *)v33 + v32) = 10;
      v36 = (_QWORD *)*a2;
      a2[1] = v34;
      *((_BYTE *)v36 + v32 + 1) = 0;
      goto LABEL_11;
    case 'r':
      v27 = a2[1];
      v28 = (_QWORD *)*a2;
      v29 = v27 + 1;
      if ( (_QWORD *)*a2 == v8 )
        v30 = 15;
      else
        v30 = a2[2];
      if ( v29 > v30 )
      {
        sub_2240BB0(a2, a2[1], 0, 0, 1);
        v28 = (_QWORD *)*a2;
      }
      *((_BYTE *)v28 + v27) = 13;
      v31 = (_QWORD *)*a2;
      a2[1] = v29;
      *((_BYTE *)v31 + v27 + 1) = 0;
      goto LABEL_11;
    case 't':
      v22 = a2[1];
      v23 = (_QWORD *)*a2;
      v24 = v22 + 1;
      if ( (_QWORD *)*a2 == v8 )
        v25 = 15;
      else
        v25 = a2[2];
      if ( v24 > v25 )
      {
        sub_2240BB0(a2, a2[1], 0, 0, 1);
        v23 = (_QWORD *)*a2;
      }
      *((_BYTE *)v23 + v22) = 9;
      v26 = (_QWORD *)*a2;
      a2[1] = v24;
      *((_BYTE *)v26 + v22 + 1) = 0;
      goto LABEL_11;
    case 'u':
      if ( v6 + 1 == v5 )
      {
        LOBYTE(v72) = 0;
        v16 = 0;
LABEL_66:
        HIBYTE(v72) = 0;
LABEL_67:
        v73 = 0;
        v17 = 0;
        goto LABEL_29;
      }
      a1[3] = (__int64)(v6 + 2);
      v16 = v6[1];
      LOBYTE(v72) = v16;
      if ( v5 == v6 + 2 )
        goto LABEL_66;
      a1[3] = (__int64)(v6 + 3);
      HIBYTE(v72) = v6[2];
      if ( v5 == v6 + 3 )
        goto LABEL_67;
      a1[3] = (__int64)(v6 + 4);
      v73 = v6[3];
      if ( v5 == v6 + 4 )
      {
        v17 = 0;
      }
      else
      {
        a1[3] = (__int64)(v6 + 5);
        v17 = v6[4];
      }
LABEL_29:
      v74 = v17;
      v18 = 0;
      for ( i = &v72; ; v16 = *(_BYTE *)i )
      {
        v66 = i;
        if ( !isxdigit(v16) )
          break;
        v21 = (v16 & 0xDF) - 55;
        if ( v16 < 0x3Au )
          v21 = v16 - 48;
        i = (__int16 *)((char *)i + 1);
        v18 = v21 | (16 * v18);
        if ( v75 == (char *)v66 + 1 )
          goto LABEL_61;
      }
      if ( (unsigned __int8)sub_C68D40(a1, (__int64)"Invalid \\u escape sequence") )
      {
LABEL_61:
        if ( (unsigned __int16)(v18 + 10240) > 0x7FFu )
        {
LABEL_62:
          sub_C699F0(v18, a2);
          goto LABEL_11;
        }
        if ( v18 > 0xDBFFu )
        {
          v63 = 0x3FFFFFFFFFFFFFFFLL - a2[1];
          v68 = -16401;
          v69 = -67;
          if ( v63 <= 2 )
            goto LABEL_99;
          sub_2241490(a2, &v68, 3, v20);
        }
        else
        {
          while ( 1 )
          {
            v48 = (_BYTE *)a1[3];
            v49 = a1[4];
            v50 = v48 + 2;
            if ( (unsigned __int64)(v48 + 2) > v49 || *v48 != 92 || v48[1] != 117 )
              break;
            if ( v50 == (_BYTE *)v49 )
            {
              a1[3] = v49;
              v51 = 0;
            }
            else
            {
              a1[3] = (__int64)(v48 + 3);
              v51 = v48[2];
            }
            v52 = (char *)a1[3];
            LOBYTE(v72) = v51;
            v53 = 0;
            if ( (char *)v49 != v52 )
            {
              a1[3] = (__int64)(v52 + 1);
              v53 = *v52++;
            }
            HIBYTE(v72) = v53;
            v54 = 0;
            if ( (char *)v49 != v52 )
            {
              a1[3] = (__int64)(v52 + 1);
              v54 = *v52++;
            }
            v73 = v54;
            v55 = 0;
            if ( (char *)v49 != v52 )
            {
              a1[3] = (__int64)(v52 + 1);
              v55 = *v52;
            }
            v74 = v55;
            v56 = &v72;
            v57 = v75;
            v58 = 0;
            while ( 1 )
            {
              v64 = v56;
              v65 = v57;
              v67 = v51;
              v59 = isxdigit(v51);
              v47 = v67;
              v57 = v65;
              if ( !v59 )
                break;
              v60 = v67 - 48;
              if ( v67 >= 0x3Au )
                v60 = (v67 & 0xDF) - 55;
              v56 = (__int16 *)((char *)v56 + 1);
              v58 = v60 | (16 * v58);
              if ( v65 == (char *)v64 + 1 )
                goto LABEL_91;
              v51 = *(_BYTE *)v56;
            }
            if ( !(unsigned __int8)sub_C68D40(a1, (__int64)"Invalid \\u escape sequence") )
              return 0;
LABEL_91:
            if ( (unsigned __int16)(v58 + 9216) <= 0x3FFu )
            {
              sub_C699F0(((v18 - 55296) << 10) | (v58 - 56320) | 0x10000, a2);
              goto LABEL_11;
            }
            v62 = 0x3FFFFFFFFFFFFFFFLL - a2[1];
            v72 = -16401;
            v73 = -67;
            if ( v62 <= 2 )
              goto LABEL_99;
            v18 = v58;
            sub_2241490(a2, &v72, 3, v47);
            if ( (unsigned __int16)(v58 + 10240) > 0x7FFu )
              goto LABEL_62;
          }
          v61 = 0x3FFFFFFFFFFFFFFFLL - a2[1];
          v70 = -16401;
          v71 = -67;
          if ( v61 <= 2 )
LABEL_99:
            sub_4262D8((__int64)"basic_string::append");
          sub_2241490(a2, &v70, 3, v50);
        }
        goto LABEL_11;
      }
      return 0;
    default:
      goto LABEL_22;
  }
}
