// Function: sub_EAE3B0
// Address: 0xeae3b0
//
__int64 __fastcall sub_EAE3B0(_QWORD *a1, _QWORD *a2)
{
  _DWORD *v3; // rax
  unsigned int v4; // r15d
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned int v8; // r12d
  __int64 v9; // r15
  char *v10; // rsi
  char v11; // al
  unsigned int v12; // eax
  __int64 v13; // rsi
  _QWORD *v14; // rdx
  unsigned __int64 v15; // r11
  unsigned __int64 v16; // rcx
  _QWORD *v17; // rax
  int v18; // edx
  const char *v19; // rax
  __int64 v20; // rax
  char v21; // al
  unsigned int v22; // edx
  unsigned int v23; // eax
  unsigned int v24; // ecx
  unsigned int v25; // r12d
  unsigned int v26; // ecx
  char v27; // al
  __int64 v28; // rsi
  _QWORD *v29; // rdx
  unsigned __int64 v30; // r11
  unsigned __int64 v31; // rcx
  _QWORD *v32; // rax
  __int64 v33; // rsi
  char v34; // r12
  _QWORD *v35; // rdx
  unsigned __int64 v36; // r11
  unsigned __int64 v37; // rcx
  _QWORD *v38; // rdx
  __int64 v39; // rsi
  _QWORD *v40; // rdx
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rcx
  __int64 v43; // rsi
  _QWORD *v44; // rdx
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rcx
  __int64 v47; // rsi
  _QWORD *v48; // rdx
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  _QWORD *v52; // rdx
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rcx
  __int64 v55; // rsi
  _QWORD *v56; // rdx
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rcx
  __int64 v59; // rsi
  _QWORD *v60; // rdx
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rcx
  __int64 v63; // rsi
  _QWORD *v64; // rdx
  unsigned __int64 v65; // rax
  unsigned __int64 v66; // rcx
  char v67; // [rsp+8h] [rbp-88h]
  char v68; // [rsp+8h] [rbp-88h]
  unsigned int v69; // [rsp+8h] [rbp-88h]
  unsigned __int64 v70; // [rsp+10h] [rbp-80h]
  _QWORD *v71; // [rsp+20h] [rbp-70h]
  int v72; // [rsp+2Ch] [rbp-64h]
  _QWORD v73[4]; // [rsp+30h] [rbp-60h] BYREF
  char v74; // [rsp+50h] [rbp-40h]
  char v75; // [rsp+51h] [rbp-3Fh]

  v75 = 1;
  v73[0] = "expected string";
  v74 = 3;
  v3 = (_DWORD *)sub_ECD7B0(a1);
  v4 = sub_ECE0A0(a1, *v3 != 3, v73);
  if ( (_BYTE)v4 )
    return v4;
  sub_2241130(a2, 0, a2[1], byte_3F871B3, 0);
  v6 = sub_ECD7B0(a1);
  v7 = *(_QWORD *)(v6 + 16);
  if ( v7 <= 1 || (v70 = v7 - 2, v72 = v7 - 2, (_DWORD)v7 == 2) )
  {
LABEL_4:
    sub_EABFE0((__int64)a1);
    return v4;
  }
  v8 = 0;
  v9 = *(_QWORD *)(v6 + 8) + 1LL;
  v71 = a2 + 2;
  while ( 1 )
  {
    v10 = (char *)(v9 + v8);
    v11 = *v10;
    if ( *v10 != 92 )
    {
      if ( v11 == 13 || v11 == 10 )
      {
        if ( v8 && v11 == 10 && *(_BYTE *)(v9 + v8 - 1) == 13 )
        {
LABEL_19:
          ++v8;
          goto LABEL_20;
        }
        v75 = 1;
        v73[0] = "unterminated string; newline inserted";
        v74 = 3;
        v12 = sub_EA8060(a1, (unsigned __int64)v10, (__int64)v73, 0, 0);
        if ( (_BYTE)v12 )
          return v12;
        v11 = *(_BYTE *)(v9 + v8);
      }
      v13 = a2[1];
      v14 = (_QWORD *)*a2;
      v15 = v13 + 1;
      if ( (_QWORD *)*a2 == v71 )
        v16 = 15;
      else
        v16 = a2[2];
      if ( v15 > v16 )
      {
        v67 = v11;
        sub_2240BB0(a2, v13, 0, 0, 1);
        v14 = (_QWORD *)*a2;
        v15 = v13 + 1;
        v11 = v67;
      }
      *((_BYTE *)v14 + v13) = v11;
      v17 = (_QWORD *)*a2;
      a2[1] = v15;
      *((_BYTE *)v17 + v13 + 1) = 0;
      goto LABEL_19;
    }
    v20 = v8 + 1;
    if ( (_DWORD)v20 == v72 )
      break;
    v21 = *(_BYTE *)(v9 + v20);
    if ( (v21 & 0xDF) == 0x58 )
    {
      v8 += 2;
      if ( v8 >= v70 || (v18 = (__int16)word_3F64060[*(unsigned __int8 *)(v9 + v8)], v18 == -1) )
      {
        v75 = 1;
        v19 = "invalid hexadecimal escape sequence";
        goto LABEL_25;
      }
      v27 = 0;
      while ( v18 != -1 )
      {
        v27 = v18 + 16 * v27;
        if ( ++v8 >= v70 )
          break;
        v18 = (__int16)word_3F64060[*(unsigned __int8 *)(v9 + v8)];
      }
      v28 = a2[1];
      v29 = (_QWORD *)*a2;
      v30 = v28 + 1;
      if ( (_QWORD *)*a2 == v71 )
        v31 = 15;
      else
        v31 = a2[2];
      if ( v30 > v31 )
      {
        v68 = v27;
        sub_2240BB0(a2, v28, 0, 0, 1);
        v29 = (_QWORD *)*a2;
        v30 = v28 + 1;
        v27 = v68;
      }
      *((_BYTE *)v29 + v28) = v27;
      v32 = (_QWORD *)*a2;
      a2[1] = v30;
      *((_BYTE *)v32 + v28 + 1) = 0;
    }
    else
    {
      v22 = v21 - 48;
      if ( v22 <= 7 )
      {
        v23 = v8 + 2;
        if ( v8 + 2 != v72 )
        {
          v24 = *(char *)(v9 + v23) - 48;
          if ( v24 <= 7 )
          {
            v25 = v8 + 3;
            v22 = v24 + 8 * v22;
            if ( v25 != v72 )
            {
              v26 = *(char *)(v9 + v25) - 48;
              if ( v26 <= 7 )
              {
                v22 = v26 + 8 * v22;
                v23 = v25;
              }
            }
            if ( v22 > 0xFF )
            {
              v75 = 1;
              v19 = "invalid octal escape sequence (out of range)";
              goto LABEL_25;
            }
            ++v23;
          }
        }
        v33 = a2[1];
        v34 = v22;
        v35 = (_QWORD *)*a2;
        v36 = v33 + 1;
        if ( (_QWORD *)*a2 == v71 )
          v37 = 15;
        else
          v37 = a2[2];
        if ( v36 > v37 )
        {
          v69 = v23;
          sub_2240BB0(a2, v33, 0, 0, 1);
          v35 = (_QWORD *)*a2;
          v36 = v33 + 1;
          v23 = v69;
        }
        *((_BYTE *)v35 + v33) = v34;
        v38 = (_QWORD *)*a2;
        v8 = v23;
        a2[1] = v36;
        *((_BYTE *)v38 + v33 + 1) = 0;
      }
      else
      {
        if ( v21 != 34 )
        {
          switch ( v21 )
          {
            case '\\':
              v51 = a2[1];
              v52 = (_QWORD *)*a2;
              v53 = v51 + 1;
              if ( (_QWORD *)*a2 == v71 )
                v54 = 15;
              else
                v54 = a2[2];
              if ( v53 > v54 )
              {
                sub_2240BB0(a2, v51, 0, 0, 1);
                v52 = (_QWORD *)*a2;
                v53 = v51 + 1;
              }
              *((_BYTE *)v52 + v51) = 92;
              v8 += 2;
              a2[1] = v53;
              *(_BYTE *)(*a2 + v51 + 1) = 0;
              goto LABEL_20;
            case 'b':
              v47 = a2[1];
              v48 = (_QWORD *)*a2;
              v49 = v47 + 1;
              if ( (_QWORD *)*a2 == v71 )
                v50 = 15;
              else
                v50 = a2[2];
              if ( v49 > v50 )
              {
                sub_2240BB0(a2, v47, 0, 0, 1);
                v48 = (_QWORD *)*a2;
                v49 = v47 + 1;
              }
              *((_BYTE *)v48 + v47) = 8;
              v8 += 2;
              a2[1] = v49;
              *(_BYTE *)(*a2 + v47 + 1) = 0;
              goto LABEL_20;
            case 'f':
              v43 = a2[1];
              v44 = (_QWORD *)*a2;
              v45 = v43 + 1;
              if ( (_QWORD *)*a2 == v71 )
                v46 = 15;
              else
                v46 = a2[2];
              if ( v45 > v46 )
              {
                sub_2240BB0(a2, v43, 0, 0, 1);
                v44 = (_QWORD *)*a2;
                v45 = v43 + 1;
              }
              *((_BYTE *)v44 + v43) = 12;
              v8 += 2;
              a2[1] = v45;
              *(_BYTE *)(*a2 + v43 + 1) = 0;
              goto LABEL_20;
            case 'n':
              v39 = a2[1];
              v40 = (_QWORD *)*a2;
              v41 = v39 + 1;
              if ( (_QWORD *)*a2 == v71 )
                v42 = 15;
              else
                v42 = a2[2];
              if ( v41 > v42 )
              {
                sub_2240BB0(a2, v39, 0, 0, 1);
                v40 = (_QWORD *)*a2;
                v41 = v39 + 1;
              }
              *((_BYTE *)v40 + v39) = 10;
              v8 += 2;
              a2[1] = v41;
              *(_BYTE *)(*a2 + v39 + 1) = 0;
              goto LABEL_20;
            case 'r':
              v59 = a2[1];
              v60 = (_QWORD *)*a2;
              v61 = v59 + 1;
              if ( (_QWORD *)*a2 == v71 )
                v62 = 15;
              else
                v62 = a2[2];
              if ( v61 > v62 )
              {
                sub_2240BB0(a2, v59, 0, 0, 1);
                v60 = (_QWORD *)*a2;
                v61 = v59 + 1;
              }
              *((_BYTE *)v60 + v59) = 13;
              v8 += 2;
              a2[1] = v61;
              *(_BYTE *)(*a2 + v59 + 1) = 0;
              goto LABEL_20;
            case 't':
              v55 = a2[1];
              v56 = (_QWORD *)*a2;
              v57 = v55 + 1;
              if ( (_QWORD *)*a2 == v71 )
                v58 = 15;
              else
                v58 = a2[2];
              if ( v57 > v58 )
              {
                sub_2240BB0(a2, v55, 0, 0, 1);
                v56 = (_QWORD *)*a2;
                v57 = v55 + 1;
              }
              *((_BYTE *)v56 + v55) = 9;
              v8 += 2;
              a2[1] = v57;
              *(_BYTE *)(*a2 + v55 + 1) = 0;
              goto LABEL_20;
            default:
              v75 = 1;
              v19 = "invalid escape sequence (unrecognized character)";
              goto LABEL_25;
          }
        }
        v63 = a2[1];
        v64 = (_QWORD *)*a2;
        v65 = v63 + 1;
        if ( (_QWORD *)*a2 == v71 )
          v66 = 15;
        else
          v66 = a2[2];
        if ( v65 > v66 )
        {
          sub_2240BB0(a2, v63, 0, 0, 1);
          v64 = (_QWORD *)*a2;
          v65 = v63 + 1;
        }
        *((_BYTE *)v64 + v63) = 34;
        v8 += 2;
        a2[1] = v65;
        *(_BYTE *)(*a2 + v63 + 1) = 0;
      }
    }
LABEL_20:
    if ( v8 == v72 )
    {
      v4 = 0;
      goto LABEL_4;
    }
  }
  v75 = 1;
  v19 = "unexpected backslash at end of string";
LABEL_25:
  v73[0] = v19;
  v74 = 3;
  return (unsigned int)sub_ECE0E0(a1, v73, 0, 0);
}
