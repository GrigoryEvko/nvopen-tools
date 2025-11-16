// Function: sub_27E0610
// Address: 0x27e0610
//
__int64 __fastcall sub_27E0610(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r13d
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  __int64 i; // r15
  __int64 v9; // rdx
  char *v10; // rdi
  char *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rdx
  char *v14; // rdx
  __int64 *v15; // r12
  _QWORD *v16; // r14
  char v17; // al
  __int64 v18; // rax
  __int64 *v19; // rax
  unsigned __int8 *v20; // r15
  __int64 v21; // r12
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // r12
  unsigned __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // r11
  __int64 v28; // rsi
  __int64 v29; // r15
  __int64 v30; // r13
  int v31; // eax
  int v32; // eax
  unsigned int v33; // ecx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rcx
  __int64 v37; // r13
  int v38; // eax
  int v39; // eax
  unsigned int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdx
  const char **v44; // r13
  const char *v45; // rsi
  unsigned __int64 v46; // rax
  int v47; // edx
  __int64 v48; // rdi
  __int64 v49; // rax
  int v50; // eax
  __m128i *v51; // rsi
  const __m128i *v52; // r8
  __m128i *v53; // rsi
  __int64 v54; // rcx
  __m128i *v55; // r8
  unsigned __int64 v56; // r9
  unsigned __int64 v57; // rax
  __int64 v58; // r14
  int v59; // eax
  __m128i *v60; // r13
  unsigned int v61; // r15d
  __int64 v62; // rax
  const __m128i *v63; // rsi
  __int64 v64; // rsi
  unsigned __int8 *v65; // rsi
  _QWORD *v66; // rax
  unsigned __int64 v67; // [rsp+8h] [rbp-98h]
  __int64 v68; // [rsp+10h] [rbp-90h]
  __int64 v69; // [rsp+10h] [rbp-90h]
  __int64 v70; // [rsp+10h] [rbp-90h]
  __int64 v71; // [rsp+10h] [rbp-90h]
  __int64 v72; // [rsp+10h] [rbp-90h]
  __int64 v73; // [rsp+10h] [rbp-90h]
  __int64 v74; // [rsp+10h] [rbp-90h]
  __int64 v75; // [rsp+20h] [rbp-80h]
  int v76; // [rsp+20h] [rbp-80h]
  __m128i v78; // [rsp+30h] [rbp-70h] BYREF
  char *v79; // [rsp+40h] [rbp-60h] BYREF
  __m128i *v80; // [rsp+48h] [rbp-58h]
  const __m128i *v81; // [rsp+50h] [rbp-50h]
  __int16 v82; // [rsp+60h] [rbp-40h]

  v3 = sub_B2D610(*(_QWORD *)(a2 + 72), 59);
  if ( (_BYTE)v3 )
    return 0;
  v4 = v3;
  if ( *(_BYTE *)(a1 + 124) )
  {
    v6 = *(_QWORD **)(a1 + 104);
    v7 = &v6[*(unsigned int *)(a1 + 116)];
    if ( v6 != v7 )
    {
      while ( a2 != *v6 )
      {
        if ( v7 == ++v6 )
          goto LABEL_8;
      }
      return 0;
    }
  }
  else if ( sub_C8CA60(a1 + 96, a2) )
  {
    return 0;
  }
LABEL_8:
  for ( i = *(_QWORD *)(a2 + 56); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      goto LABEL_126;
    if ( *(_BYTE *)(i - 24) != 84 )
      return v4;
    v9 = 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF);
    if ( (*(_BYTE *)(i - 17) & 0x40) != 0 )
    {
      v11 = *(char **)(i - 32);
      v10 = &v11[v9];
    }
    else
    {
      v10 = (char *)(i - 24);
      v11 = (char *)(i - 24 - v9);
    }
    v12 = v9 >> 5;
    v13 = v9 >> 7;
    if ( v13 )
    {
      v14 = &v11[128 * v13];
      while ( 1 )
      {
        if ( **(_BYTE **)v11 == 17 )
          goto LABEL_20;
        if ( **((_BYTE **)v11 + 4) == 17 )
        {
          v11 += 32;
          goto LABEL_20;
        }
        if ( **((_BYTE **)v11 + 8) == 17 )
        {
          v11 += 64;
          goto LABEL_20;
        }
        if ( **((_BYTE **)v11 + 12) == 17 )
          break;
        v11 += 128;
        if ( v14 == v11 )
        {
          v12 = (v10 - v11) >> 5;
          goto LABEL_90;
        }
      }
      v11 += 96;
      goto LABEL_20;
    }
LABEL_90:
    if ( v12 == 2 )
      goto LABEL_100;
    if ( v12 == 3 )
    {
      if ( **(_BYTE **)v11 == 17 )
        goto LABEL_20;
      v11 += 32;
LABEL_100:
      if ( **(_BYTE **)v11 == 17 )
        goto LABEL_20;
      v11 += 32;
      goto LABEL_93;
    }
    if ( v12 != 1 )
      continue;
LABEL_93:
    if ( **(_BYTE **)v11 != 17 )
      continue;
LABEL_20:
    if ( v11 != v10 )
    {
      v15 = *(__int64 **)(i - 8);
      if ( v15 )
        break;
    }
LABEL_86:
    ;
  }
  while ( 1 )
  {
    v16 = (_QWORD *)v15[3];
    v17 = *(_BYTE *)v16;
    if ( *(_BYTE *)v16 <= 0x1Cu )
      goto LABEL_24;
    if ( v17 == 82 )
      break;
    if ( v17 == 86 && a2 == v16[5] )
    {
      if ( sub_27DBB30(v15[3], *v15) )
        goto LABEL_34;
      v15 = (__int64 *)v15[1];
      if ( !v15 )
        goto LABEL_86;
    }
    else
    {
LABEL_24:
      v15 = (__int64 *)v15[1];
      if ( !v15 )
        goto LABEL_86;
    }
  }
  if ( a2 != v16[5] )
    goto LABEL_24;
  v18 = v16[2];
  if ( !v18 )
    goto LABEL_24;
  if ( *(_QWORD *)(v18 + 8) )
    goto LABEL_24;
  if ( *(_BYTE *)v16[4 * (1 - (unsigned int)sub_BD2910((__int64)v15)) - 8] != 17 )
    goto LABEL_24;
  v19 = (__int64 *)v16[2];
  v16 = (_QWORD *)v19[3];
  if ( *(_BYTE *)v16 != 86 || a2 != v16[5] || !sub_27DBB30(v19[3], *v19) )
    goto LABEL_24;
LABEL_34:
  v20 = (unsigned __int8 *)*(v16 - 12);
  v21 = (__int64)v20;
  if ( !sub_98ED60(v20, 0, (__int64)v16, 0, 0) )
  {
    v79 = "cond.fr";
    v82 = 259;
    v66 = sub_BD2C40(72, unk_3F10A14);
    v21 = (__int64)v66;
    if ( v66 )
      sub_B549F0((__int64)v66, (__int64)v20, (__int64)&v79, (__int64)(v16 + 3), 0);
  }
  v22 = sub_BC89C0((__int64)v16);
  v23 = sub_F38250(v21, v16 + 3, 0, 0, v22, 0, 0, 0);
  v24 = v16[5];
  v25 = v23;
  v75 = *(_QWORD *)(v23 + 40);
  v82 = 257;
  v68 = v16[1];
  v26 = sub_BD2DA0(80);
  v27 = v26;
  if ( v26 )
  {
    v28 = v68;
    v69 = v26;
    sub_B44260(v26, v28, 55, 0x8000000u, (__int64)(v16 + 3), 0);
    *(_DWORD *)(v69 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v69, (const char **)&v79);
    sub_BD2A10(v69, *(_DWORD *)(v69 + 72), 1);
    v27 = v69;
  }
  v29 = *(_QWORD *)(v25 + 40);
  v30 = *(v16 - 8);
  v31 = *(_DWORD *)(v27 + 4) & 0x7FFFFFF;
  if ( v31 == *(_DWORD *)(v27 + 72) )
  {
    v74 = v27;
    sub_B48D90(v27);
    v27 = v74;
    v31 = *(_DWORD *)(v74 + 4) & 0x7FFFFFF;
  }
  v32 = (v31 + 1) & 0x7FFFFFF;
  v33 = v32 | *(_DWORD *)(v27 + 4) & 0xF8000000;
  v34 = *(_QWORD *)(v27 - 8) + 32LL * (unsigned int)(v32 - 1);
  *(_DWORD *)(v27 + 4) = v33;
  if ( *(_QWORD *)v34 )
  {
    v35 = *(_QWORD *)(v34 + 8);
    **(_QWORD **)(v34 + 16) = v35;
    if ( v35 )
      *(_QWORD *)(v35 + 16) = *(_QWORD *)(v34 + 16);
  }
  *(_QWORD *)v34 = v30;
  if ( v30 )
  {
    v36 = *(_QWORD *)(v30 + 16);
    *(_QWORD *)(v34 + 8) = v36;
    if ( v36 )
      *(_QWORD *)(v36 + 16) = v34 + 8;
    *(_QWORD *)(v34 + 16) = v30 + 16;
    *(_QWORD *)(v30 + 16) = v34;
  }
  *(_QWORD *)(*(_QWORD *)(v27 - 8) + 32LL * *(unsigned int *)(v27 + 72)
                                   + 8LL * ((*(_DWORD *)(v27 + 4) & 0x7FFFFFFu) - 1)) = v29;
  v37 = *(v16 - 4);
  v38 = *(_DWORD *)(v27 + 4) & 0x7FFFFFF;
  if ( v38 == *(_DWORD *)(v27 + 72) )
  {
    v73 = v27;
    sub_B48D90(v27);
    v27 = v73;
    v38 = *(_DWORD *)(v73 + 4) & 0x7FFFFFF;
  }
  v39 = (v38 + 1) & 0x7FFFFFF;
  v40 = v39 | *(_DWORD *)(v27 + 4) & 0xF8000000;
  v41 = *(_QWORD *)(v27 - 8) + 32LL * (unsigned int)(v39 - 1);
  *(_DWORD *)(v27 + 4) = v40;
  if ( *(_QWORD *)v41 )
  {
    v42 = *(_QWORD *)(v41 + 8);
    **(_QWORD **)(v41 + 16) = v42;
    if ( v42 )
      *(_QWORD *)(v42 + 16) = *(_QWORD *)(v41 + 16);
  }
  *(_QWORD *)v41 = v37;
  if ( v37 )
  {
    v43 = *(_QWORD *)(v37 + 16);
    *(_QWORD *)(v41 + 8) = v43;
    if ( v43 )
      *(_QWORD *)(v43 + 16) = v41 + 8;
    *(_QWORD *)(v41 + 16) = v37 + 16;
    *(_QWORD *)(v37 + 16) = v41;
  }
  v44 = (const char **)(v27 + 48);
  *(_QWORD *)(*(_QWORD *)(v27 - 8) + 32LL * *(unsigned int *)(v27 + 72)
                                   + 8LL * ((*(_DWORD *)(v27 + 4) & 0x7FFFFFFu) - 1)) = a2;
  v45 = (const char *)v16[6];
  v79 = (char *)v45;
  if ( !v45 )
  {
    if ( v44 == (const char **)&v79 )
      goto LABEL_59;
    v64 = *(_QWORD *)(v27 + 48);
    if ( !v64 )
      goto LABEL_59;
LABEL_111:
    v71 = v27;
    sub_B91220((__int64)v44, v64);
    v27 = v71;
    goto LABEL_112;
  }
  v70 = v27;
  sub_B96E90((__int64)&v79, (__int64)v45, 1);
  v27 = v70;
  if ( v44 == (const char **)&v79 )
  {
    if ( v79 )
    {
      sub_B91220((__int64)&v79, (__int64)v79);
      v27 = v70;
    }
    goto LABEL_59;
  }
  v64 = *(_QWORD *)(v70 + 48);
  if ( v64 )
    goto LABEL_111;
LABEL_112:
  v65 = (unsigned __int8 *)v79;
  *(_QWORD *)(v27 + 48) = v79;
  if ( v65 )
  {
    v72 = v27;
    sub_B976B0((__int64)&v79, v65, (__int64)v44);
    v27 = v72;
  }
LABEL_59:
  sub_BD84D0((__int64)v16, v27);
  sub_B43D60(v16);
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v46 = *(_QWORD *)(v24 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 + 48 == v46 )
  {
    v48 = 0;
    goto LABEL_63;
  }
  if ( !v46 )
LABEL_126:
    BUG();
  v47 = *(unsigned __int8 *)(v46 - 24);
  v48 = 0;
  v49 = v46 - 24;
  if ( (unsigned int)(v47 - 30) < 0xB )
    v48 = v49;
LABEL_63:
  v50 = sub_B46E30(v48);
  sub_F58D10((const __m128i **)&v79, (unsigned int)(2 * v50 + 3));
  v51 = v80;
  v78.m128i_i64[0] = a2;
  v52 = v81;
  v78.m128i_i64[1] = v24 & 0xFFFFFFFFFFFFFFFBLL;
  if ( v80 == v81 )
  {
    sub_F38BA0((const __m128i **)&v79, v80, &v78);
    v78.m128i_i64[0] = a2;
    v53 = v80;
    v78.m128i_i64[1] = v75 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v81 == v80 )
    {
      v52 = v80;
      goto LABEL_122;
    }
    if ( v80 )
      goto LABEL_67;
LABEL_68:
    v80 = v53 + 1;
  }
  else
  {
    if ( v80 )
    {
      *v80 = _mm_loadu_si128(&v78);
      v51 = v80;
      v52 = v81;
    }
    v53 = v51 + 1;
    v78.m128i_i64[0] = a2;
    v80 = v53;
    v78.m128i_i64[1] = v75 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v53 != v52 )
    {
LABEL_67:
      *v53 = _mm_loadu_si128(&v78);
      v53 = v80;
      goto LABEL_68;
    }
LABEL_122:
    sub_F38BA0((const __m128i **)&v79, v52, &v78);
  }
  v78.m128i_i64[1] = v24 & 0xFFFFFFFFFFFFFFFBLL;
  v78.m128i_i64[0] = v75;
  sub_27E05D0((__int64)&v79, &v78);
  v57 = *(_QWORD *)(v24 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 + 48 == v57 )
    goto LABEL_105;
  if ( !v57 )
    BUG();
  v58 = v57 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v57 - 24) - 30 > 0xA )
  {
LABEL_105:
    v60 = v80;
  }
  else
  {
    v59 = sub_B46E30(v58);
    v60 = v80;
    v76 = v59;
    if ( v59 )
    {
      v61 = 0;
      while ( 1 )
      {
        v62 = sub_B46EC0(v58, v61);
        v63 = v81;
        v78.m128i_i64[0] = a2;
        v78.m128i_i64[1] = v62 | 4;
        v56 = v62 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v81 != v60 )
          break;
        v67 = v62 & 0xFFFFFFFFFFFFFFFBLL;
        sub_F38BA0((const __m128i **)&v79, v60, &v78);
        v56 = v67;
        v78.m128i_i64[0] = v24;
        v55 = v80;
        v78.m128i_i64[1] = v67;
        if ( v81 == v80 )
        {
          v63 = v80;
LABEL_81:
          sub_F38BA0((const __m128i **)&v79, v63, &v78);
          v60 = v80;
          goto LABEL_76;
        }
        if ( v80 )
          goto LABEL_74;
LABEL_75:
        v60 = v55 + 1;
        v80 = v55 + 1;
LABEL_76:
        if ( v76 == ++v61 )
          goto LABEL_106;
      }
      if ( v60 )
      {
        *v60 = _mm_loadu_si128(&v78);
        v63 = v81;
        v60 = v80;
      }
      v55 = v60 + 1;
      v78.m128i_i64[0] = v24;
      v80 = v60 + 1;
      v78.m128i_i64[1] = v62 & 0xFFFFFFFFFFFFFFFBLL;
      if ( &v60[1] == v63 )
        goto LABEL_81;
LABEL_74:
      *v55 = _mm_loadu_si128(&v78);
      v55 = v80;
      goto LABEL_75;
    }
  }
LABEL_106:
  sub_FFDB80(*(_QWORD *)(a1 + 48), (unsigned __int64 *)v79, ((char *)v60 - v79) >> 4, v54, (__int64)v55, v56);
  if ( v79 )
    j_j___libc_free_0((unsigned __int64)v79);
  return 1;
}
