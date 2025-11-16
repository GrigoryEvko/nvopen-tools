// Function: sub_C42160
// Address: 0xc42160
//
__int64 __fastcall sub_C42160(__int64 a1, __int64 *a2, unsigned __int8 *a3, __int64 a4, char a5)
{
  unsigned __int8 *v5; // r12
  unsigned __int64 *v7; // rdi
  unsigned __int8 *v8; // r13
  unsigned __int8 *v9; // rbx
  unsigned __int64 v10; // rsi
  __int64 v11; // r8
  unsigned __int8 *v12; // r9
  signed __int64 v13; // rdx
  char *v14; // rcx
  char v15; // al
  unsigned __int8 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rcx
  int v19; // esi
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  char v22; // al
  int v23; // ebx
  char v24; // al
  __int64 v26; // rdi
  __int64 v27; // rax
  unsigned int v28; // r8d
  __int64 v29; // r14
  unsigned __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  int v34; // eax
  const char *v35; // rax
  unsigned int v36; // eax
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  char v40; // al
  unsigned int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned int v49; // eax
  __int64 v50; // rdx
  __int64 v51; // r12
  unsigned int v52; // r13d
  __int64 v53; // rax
  __int64 v54; // rbx
  unsigned int v55; // r12d
  unsigned int v56; // r13d
  __int64 v57; // [rsp-8h] [rbp-B8h]
  int v58; // [rsp+8h] [rbp-A8h]
  int v60; // [rsp+10h] [rbp-A0h]
  __int64 v61; // [rsp+10h] [rbp-A0h]
  __int64 v62; // [rsp+10h] [rbp-A0h]
  unsigned int v63; // [rsp+10h] [rbp-A0h]
  __int64 v64; // [rsp+10h] [rbp-A0h]
  char *v65; // [rsp+18h] [rbp-98h]
  unsigned int v66; // [rsp+18h] [rbp-98h]
  __int64 v69; // [rsp+28h] [rbp-88h]
  unsigned __int64 v70; // [rsp+28h] [rbp-88h]
  __int64 v71; // [rsp+28h] [rbp-88h]
  __int64 v72; // [rsp+28h] [rbp-88h]
  __int64 v73; // [rsp+28h] [rbp-88h]
  __int64 v74; // [rsp+28h] [rbp-88h]
  unsigned int v75; // [rsp+28h] [rbp-88h]
  unsigned int v76; // [rsp+28h] [rbp-88h]
  unsigned int v77; // [rsp+28h] [rbp-88h]
  __int64 v78; // [rsp+30h] [rbp-80h]
  unsigned __int8 *v79; // [rsp+38h] [rbp-78h] BYREF
  unsigned __int64 v80; // [rsp+40h] [rbp-70h] BYREF
  char v81; // [rsp+48h] [rbp-68h]
  _QWORD v82[4]; // [rsp+50h] [rbp-60h] BYREF
  char v83; // [rsp+70h] [rbp-40h]
  char v84; // [rsp+71h] [rbp-3Fh]

  v7 = &v80;
  v8 = a3;
  v9 = &a3[a4];
  v10 = (unsigned __int64)a3;
  v79 = &a3[a4];
  sub_C32010((__int64)&v80, a3, &a3[a4], &v79);
  v12 = a3;
  v13 = v81 & 1;
  v14 = (char *)(unsigned int)(2 * v13);
  v15 = (2 * v13) | v81 & 0xFD;
  v81 = v15;
  if ( (_BYTE)v13 )
  {
    v81 = v15 & 0xFD;
    v20 = v80;
    v80 = 0;
    v78 = v20 | 1;
    goto LABEL_20;
  }
  v16 = (unsigned __int8 *)v80;
  v65 = (char *)v80;
  if ( v9 == (unsigned __int8 *)v80 )
  {
    v58 = 0;
    v5 = v9;
    v60 = 0;
    goto LABEL_54;
  }
  while ( 1 )
  {
    v17 = *v16;
    if ( (_BYTE)v17 != 46 )
    {
      v13 = (unsigned int)((char)v17 - 48);
      if ( (unsigned int)v13 > 9 )
        break;
      goto LABEL_5;
    }
    if ( v9 != v79 )
    {
      v84 = 1;
      v82[0] = "String contains multiple dots";
      v83 = 3;
      v41 = sub_C63BB0(&v80, v10, v13, v17);
      v62 = v42;
      v71 = v41;
      v43 = sub_22077B0(64);
      if ( v43 )
      {
        v44 = v71;
        v10 = (unsigned __int64)v82;
        v72 = v43;
        sub_C63EB0(v43, v82, v44, v62);
        v43 = v72;
      }
      v78 = v43 | 1;
      v40 = v81;
      v13 = (v81 & 2) != 0;
      goto LABEL_48;
    }
    v5 = v16 + 1;
    v79 = v16;
    if ( v9 == v16 + 1 )
      goto LABEL_52;
    LODWORD(v17) = *++v16;
    v13 = (unsigned int)((char)v17 - 48);
    if ( (unsigned int)v13 > 9 )
      break;
LABEL_5:
    if ( v9 == ++v16 )
      goto LABEL_51;
  }
  if ( v9 == v16 )
  {
LABEL_51:
    v5 = v16;
    v16 = v79;
LABEL_52:
    v60 = 0;
    if ( a3 != v5 )
    {
      do
LABEL_60:
        --v5;
      while ( a3 != v5 && (*v5 == 48 || a3 != v5 && *v5 == 46) );
    }
    v10 = v80;
    v60 += (_DWORD)v16 - ((v5 < v16) + (_DWORD)v5);
    v14 = (char *)&v5[-v80];
    v13 = v60 + (_DWORD)v5 - (_DWORD)v80 - (unsigned int)(v5 > v16 && v80 < (unsigned __int64)v16);
    v58 = v60 + (_DWORD)v5 - v80 - (v5 > v16 && v80 < (unsigned __int64)v16);
    goto LABEL_54;
  }
  v18 = (unsigned int)v17 & 0xFFFFFFDF;
  if ( (_BYTE)v18 != 69 )
  {
    v84 = 1;
    v35 = "Invalid character in significand";
    goto LABEL_45;
  }
  if ( a3 == v16 )
  {
LABEL_65:
    v84 = 1;
    v35 = "Significand has no digits";
LABEL_45:
    v82[0] = v35;
    v83 = 3;
    v36 = sub_C63BB0(v7, v10, v13, v18);
    v61 = v37;
    v69 = v36;
    v38 = sub_22077B0(64);
    if ( v38 )
    {
      v39 = v69;
      v10 = (unsigned __int64)v82;
      v70 = v38;
      sub_C63EB0(v38, v82, v39, v61);
      v38 = v70;
    }
    goto LABEL_47;
  }
  v7 = (unsigned __int64 *)v79;
  if ( v9 == v79 )
  {
    v14 = (char *)(v16 + 1);
    if ( v9 == v16 + 1 )
    {
      v60 = 0;
      goto LABEL_57;
    }
LABEL_16:
    v11 = v16[1];
    v13 = (unsigned __int8)(v11 - 43) & 0xFD;
    if ( (((_BYTE)v11 - 43) & 0xFD) != 0 )
    {
      v19 = (char)v11;
    }
    else
    {
      v14 = (char *)(v16 + 2);
      if ( v9 == v16 + 2 )
      {
        v60 = 0;
LABEL_80:
        if ( v9 != v79 )
          goto LABEL_58;
LABEL_57:
        v79 = v16;
        goto LABEL_58;
      }
      v19 = (char)v16[2];
    }
    v10 = (unsigned int)(v19 - 48);
    if ( (unsigned int)v10 > 9 )
    {
LABEL_82:
      v84 = 1;
      v82[0] = "Invalid character in exponent";
      v83 = 3;
      v45 = sub_C63BB0(v79, v10, v13, v14);
      v64 = v46;
      v73 = v45;
      v47 = sub_22077B0(64);
      if ( v47 )
      {
        v48 = v73;
        v10 = (unsigned __int64)v82;
        v74 = v47;
        sub_C63EB0(v47, v82, v48, v64);
        v47 = v74;
      }
      v38 = v47 & 0xFFFFFFFFFFFFFFFELL;
LABEL_47:
      v78 = v38 | 1;
      v40 = v81;
      v13 = (v81 & 2) != 0;
LABEL_48:
      v58 = 0;
      v60 = 0;
      if ( (_BYTE)v13 )
        sub_C420F0(&v80);
      goto LABEL_49;
    }
    do
    {
      if ( v9 == (unsigned __int8 *)++v14 )
      {
        v63 = v10;
        goto LABEL_77;
      }
      v13 = (unsigned int)(*v14 - 48);
      if ( (unsigned int)v13 > 9 )
        goto LABEL_82;
      v10 = (unsigned int)(v13 + 10 * v10);
    }
    while ( (unsigned int)v10 <= 0x5DBF );
    v63 = 24000;
LABEL_77:
    v10 = v63;
    v13 = -v63;
    if ( (_BYTE)v11 != 45 )
      v13 = v63;
    v60 = v13;
    goto LABEL_80;
  }
  v13 = v16 - a3;
  if ( v16 - a3 == 1 )
    goto LABEL_65;
  v14 = (char *)(v16 + 1);
  if ( v9 != v16 + 1 )
    goto LABEL_16;
  v60 = 0;
LABEL_58:
  if ( (unsigned __int8 *)v80 != v16 )
  {
    v5 = v16;
    v16 = v79;
    goto LABEL_60;
  }
  v58 = 0;
  v5 = (unsigned __int8 *)v80;
LABEL_54:
  v78 = 1;
  v40 = v81;
LABEL_49:
  if ( (v40 & 1) != 0 )
  {
LABEL_20:
    if ( v80 )
      (*(void (__fastcall **)(unsigned __int64, unsigned __int64, signed __int64, char *, __int64, unsigned __int8 *))(*(_QWORD *)v80 + 8LL))(
        v80,
        v10,
        v13,
        v14,
        v11,
        v12);
  }
  if ( (v78 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v78 & 0xFFFFFFFFFFFFFFFELL;
  }
  else if ( v9 == (unsigned __int8 *)v65 || (unsigned int)(*v65 - 48) > 9 )
  {
    v21 = *a2;
    v22 = *((_BYTE *)a2 + 20) & 0xF8 | 3;
    *((_BYTE *)a2 + 20) = v22;
    if ( *(_DWORD *)(v21 + 20) == 2 )
      *((_BYTE *)a2 + 20) = v22 & 0xF7;
    v23 = 0;
    if ( !*(_BYTE *)(v21 + 24) )
      sub_C35A40((__int64)a2, 0);
LABEL_29:
    v24 = *(_BYTE *)(a1 + 8);
    *(_DWORD *)a1 = v23;
    *(_BYTE *)(a1 + 8) = v24 & 0xFC | 2;
  }
  else
  {
    if ( v58 > 51084 )
      goto LABEL_88;
    if ( v58 < -51082 || 28738 * (v58 + 1) <= 8651 * (*(_DWORD *)(*a2 + 4) - *(_DWORD *)(*a2 + 8)) )
    {
      *((_BYTE *)a2 + 20) = *((_BYTE *)a2 + 20) & 0xF8 | 2;
      sub_C33EE0((__int64)a2);
      v23 = sub_C36450((__int64)a2, a5, 1);
      goto LABEL_29;
    }
    if ( 42039 * (v58 - 1) >= 12655 * *(_DWORD *)*a2 )
    {
LABEL_88:
      v23 = sub_C36320((__int64)a2, a5);
      goto LABEL_29;
    }
    v26 = 8LL * (((196 * ((int)v5 - (int)v65 + 1) / 0x3Bu + 64) >> 6) + 1);
    v27 = sub_2207820(v26);
    v28 = 0;
    v29 = v27;
LABEL_37:
    v30 = 0;
    v31 = 19;
    v32 = 1;
    while ( 1 )
    {
      v34 = (char)*v8;
      if ( *v8 == 46 )
      {
        v26 = (__int64)(v8 + 1);
        if ( v9 == v8 + 1 )
        {
          v56 = v28 + 1;
          v26 = v29;
          v77 = v28;
          sub_C46FF0(v29, v29, v32, v30, v28, v28 + 1, 0);
          v28 = v77;
          if ( *(_QWORD *)(v29 + 8LL * v77) )
            v28 = v56;
          if ( v9 <= v5 )
          {
            v8 = v9;
            goto LABEL_37;
          }
          v55 = v28;
          goto LABEL_101;
        }
        v34 = (char)*++v8;
      }
      v33 = (unsigned int)(v34 - 48);
      ++v8;
      if ( v33 > 9 )
        break;
      v32 *= 10;
      v30 = v33 + 10 * v30;
      if ( v8 > v5 )
      {
        v55 = v28 + 1;
        v76 = v28;
        sub_C46FF0(v29, v29, v32, v30, v28, v28 + 1, 0);
        if ( !*(_QWORD *)(v29 + 8LL * v76) )
          v55 = v76;
LABEL_101:
        *((_BYTE *)a2 + 20) = *((_BYTE *)a2 + 20) & 0xF8 | 2;
        v23 = sub_C3A080(a2, v29, v55, v60, a5);
        if ( v29 )
          j_j___libc_free_0_0(v29);
        goto LABEL_29;
      }
      v31 = (unsigned int)(v31 - 1);
      if ( !(_DWORD)v31 )
      {
        v66 = v28 + 1;
        v75 = v28;
        sub_C46FF0(v29, v29, v32, v30, v28, v28 + 1, 0);
        v26 = v57;
        v28 = v75;
        if ( *(_QWORD *)(v29 + 8LL * v75) )
          v28 = v66;
        goto LABEL_37;
      }
    }
    if ( v29 )
    {
      v26 = v29;
      j_j___libc_free_0_0(v29);
    }
    v84 = 1;
    v82[0] = "Invalid character in significand";
    v83 = 3;
    v49 = sub_C63BB0(v26, v31, v32, v30);
    v51 = v50;
    v52 = v49;
    v53 = sub_22077B0(64);
    v54 = v53;
    if ( v53 )
      sub_C63EB0(v53, v82, v52, v51);
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v54 & 0xFFFFFFFFFFFFFFFELL;
  }
  return a1;
}
