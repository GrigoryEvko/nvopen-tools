// Function: sub_29208D0
// Address: 0x29208d0
//
__int64 __fastcall sub_29208D0(__int64 a1, __int64 a2)
{
  unsigned __int8 **v2; // r12
  unsigned __int8 *v3; // rax
  __int64 v4; // rax
  _BYTE *v5; // rdi
  unsigned __int8 *v7; // r13
  __int64 v8; // rbx
  unsigned __int8 **v9; // r15
  __int64 v10; // rax
  __int64 v11; // rbx
  unsigned __int8 **v12; // rbx
  unsigned __int8 *v13; // rdi
  unsigned __int8 v14; // al
  unsigned __int8 **v15; // rdx
  unsigned __int8 *v16; // rdi
  unsigned __int8 v17; // al
  __int64 v18; // rcx
  char *v19; // rax
  char *v20; // r8
  __int64 v21; // rsi
  __int64 v22; // rcx
  char *v23; // rcx
  __int64 *v24; // r14
  const char *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // r15
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rsi
  __int64 v35; // rax
  unsigned int v36; // eax
  __int64 v37; // r8
  __int64 v38; // r12
  int v39; // edx
  __int64 v40; // rdi
  __int64 v41; // rax
  _BYTE *v42; // r14
  int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 *v47; // rdx
  unsigned __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  unsigned __int8 *v51; // rdi
  unsigned __int8 **v52; // r14
  unsigned __int8 v53; // al
  unsigned __int8 *v54; // rdi
  unsigned __int8 v55; // al
  unsigned __int8 *v56; // rdi
  unsigned __int8 v57; // al
  unsigned __int8 *v58; // rdi
  unsigned __int8 v59; // al
  __int64 v60; // r14
  unsigned __int8 **v61; // r8
  __int64 v62; // rax
  unsigned __int8 **v63; // rbx
  unsigned __int8 **v64; // r14
  unsigned __int8 *v65; // r12
  bool v66; // zf
  __int64 v67; // r8
  __int64 v68; // r9
  unsigned __int64 v69; // rdx
  const char *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rcx
  unsigned int **v73; // r14
  unsigned int v74; // ebx
  __int64 v75; // rdx
  __int64 *v76; // r14
  int v77; // eax
  __int64 v78; // rdx
  __int64 v79; // rax
  unsigned __int8 *v80; // rdi
  unsigned __int8 v81; // al
  unsigned __int8 *v82; // rdi
  unsigned __int8 v83; // al
  __int64 v84; // [rsp+8h] [rbp-E8h]
  __int64 v85; // [rsp+10h] [rbp-E0h]
  __int64 v86; // [rsp+18h] [rbp-D8h]
  __int64 v87; // [rsp+28h] [rbp-C8h]
  __int64 v88; // [rsp+28h] [rbp-C8h]
  __int64 v89; // [rsp+30h] [rbp-C0h]
  __int64 v90; // [rsp+38h] [rbp-B8h]
  unsigned __int8 *v92; // [rsp+48h] [rbp-A8h]
  __int64 v93; // [rsp+48h] [rbp-A8h]
  const char *v94; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v95; // [rsp+58h] [rbp-98h]
  char *v96; // [rsp+60h] [rbp-90h]
  __int16 v97; // [rsp+70h] [rbp-80h]
  const char *v98; // [rsp+80h] [rbp-70h] BYREF
  __int64 v99; // [rsp+88h] [rbp-68h]
  _QWORD v100[2]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v101; // [rsp+A0h] [rbp-50h]

  v2 = (unsigned __int8 **)a2;
  v3 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a2);
  if ( *v3 != 84 )
  {
    v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v5 = *(_BYTE **)(a2 - 32 * v4);
    if ( *v5 > 0x1Cu )
    {
      if ( *v5 != 60 || !sub_B4D040((__int64)v5) )
        return 0;
      v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    }
    v15 = (unsigned __int8 **)(a2 + 32 * (1 - v4));
    if ( (unsigned __int8 **)a2 == v15 )
      return 0;
    v16 = *v15;
    v7 = 0;
    v17 = **v15;
    if ( v17 > 0x1Cu )
      goto LABEL_15;
LABEL_58:
    if ( v17 != 17 )
      return 0;
    while ( 1 )
    {
      v15 += 4;
      if ( v2 == v15 )
      {
        if ( v7 )
          goto LABEL_30;
        return 0;
      }
LABEL_14:
      v16 = *v15;
      v17 = **v15;
      if ( v17 <= 0x1Cu )
        goto LABEL_58;
LABEL_15:
      if ( v17 != 84 || v7 )
        return 0;
      v18 = 32LL * (*((_DWORD *)v16 + 1) & 0x7FFFFFF);
      if ( (v16[7] & 0x40) != 0 )
      {
        v19 = (char *)*((_QWORD *)v16 - 1);
        v20 = &v19[v18];
      }
      else
      {
        v20 = (char *)v16;
        v19 = (char *)&v16[-v18];
      }
      v21 = v18 >> 5;
      v22 = v18 >> 7;
      if ( v22 )
      {
        v23 = &v19[128 * v22];
        while ( 1 )
        {
          if ( **(_BYTE **)v19 != 17 )
            goto LABEL_26;
          if ( **((_BYTE **)v19 + 4) != 17 )
          {
            v19 += 32;
            goto LABEL_26;
          }
          if ( **((_BYTE **)v19 + 8) != 17 )
          {
            v19 += 64;
            goto LABEL_26;
          }
          if ( **((_BYTE **)v19 + 12) != 17 )
            break;
          v19 += 128;
          if ( v23 == v19 )
          {
            v21 = (v20 - v19) >> 5;
            goto LABEL_112;
          }
        }
        v19 += 96;
        goto LABEL_26;
      }
LABEL_112:
      if ( v21 == 2 )
        goto LABEL_123;
      if ( v21 != 3 )
      {
        if ( v21 != 1 )
          goto LABEL_27;
LABEL_115:
        if ( **(_BYTE **)v19 == 17 )
          goto LABEL_27;
        goto LABEL_26;
      }
      if ( **(_BYTE **)v19 == 17 )
        break;
LABEL_26:
      if ( v20 != v19 )
        return 0;
LABEL_27:
      v7 = v16;
    }
    v19 += 32;
LABEL_123:
    if ( **(_BYTE **)v19 == 17 )
    {
      v19 += 32;
      goto LABEL_115;
    }
    goto LABEL_26;
  }
  v7 = v3;
  v8 = 32LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF);
  if ( (v3[7] & 0x40) != 0 )
  {
    v9 = (unsigned __int8 **)*((_QWORD *)v3 - 1);
    v92 = (unsigned __int8 *)&v9[(unsigned __int64)v8 / 8];
  }
  else
  {
    v92 = v3;
    v9 = (unsigned __int8 **)&v3[-v8];
  }
  v10 = v8 >> 5;
  v11 = v8 >> 7;
  if ( v11 )
  {
    v12 = &v9[16 * v11];
    do
    {
      v13 = *v9;
      v14 = **v9;
      if ( v14 > 0x1Cu )
      {
        if ( v14 == 60 )
        {
          if ( !sub_B4D040((__int64)v13) )
            goto LABEL_12;
        }
        else if ( v14 != 63 || v7 == sub_98ACB0(v13, 6u) )
        {
          goto LABEL_12;
        }
      }
      v51 = v9[4];
      v52 = v9 + 4;
      v53 = *v51;
      if ( *v51 > 0x1Cu )
      {
        if ( v53 == 60 )
        {
          if ( !sub_B4D040((__int64)v51) )
            goto LABEL_64;
        }
        else if ( v53 != 63 || v7 == sub_98ACB0(v51, 6u) )
        {
LABEL_64:
          v9 = v52;
          goto LABEL_12;
        }
      }
      v54 = v9[8];
      v52 = v9 + 8;
      v55 = *v54;
      if ( *v54 > 0x1Cu )
      {
        if ( v55 == 60 )
        {
          if ( !sub_B4D040((__int64)v54) )
            goto LABEL_64;
        }
        else if ( v55 != 63 || v7 == sub_98ACB0(v54, 6u) )
        {
          goto LABEL_64;
        }
      }
      v56 = v9[12];
      v52 = v9 + 12;
      v57 = *v56;
      if ( *v56 > 0x1Cu )
      {
        if ( v57 == 60 )
        {
          if ( !sub_B4D040((__int64)v56) )
            goto LABEL_64;
        }
        else if ( v57 != 63 || v7 == sub_98ACB0(v56, 6u) )
        {
          goto LABEL_64;
        }
      }
      v9 += 16;
    }
    while ( v12 != v9 );
    v10 = (v92 - (unsigned __int8 *)v9) >> 5;
  }
  if ( v10 == 2 )
  {
LABEL_130:
    v82 = *v9;
    v83 = **v9;
    if ( v83 > 0x1Cu )
    {
      if ( v83 == 60 )
      {
        if ( !sub_B4D040((__int64)v82) )
          goto LABEL_12;
      }
      else if ( v83 != 63 || v7 == sub_98ACB0(v82, 6u) )
      {
        goto LABEL_12;
      }
    }
    v9 += 4;
    goto LABEL_79;
  }
  if ( v10 == 3 )
  {
    v80 = *v9;
    v81 = **v9;
    if ( v81 > 0x1Cu )
    {
      if ( v81 == 60 )
      {
        if ( !sub_B4D040((__int64)v80) )
          goto LABEL_12;
      }
      else if ( v81 != 63 || v7 == sub_98ACB0(v80, 6u) )
      {
        goto LABEL_12;
      }
    }
    v9 += 4;
    goto LABEL_130;
  }
  if ( v10 != 1 )
    goto LABEL_13;
LABEL_79:
  v58 = *v9;
  v59 = **v9;
  if ( v59 <= 0x1Cu )
    goto LABEL_13;
  if ( v59 == 60 )
  {
    if ( sub_B4D040((__int64)v58) )
      goto LABEL_13;
  }
  else if ( v59 == 63 && v7 != sub_98ACB0(v58, 6u) )
  {
    goto LABEL_13;
  }
LABEL_12:
  if ( v9 != (unsigned __int8 **)v92 )
    return 0;
LABEL_13:
  v15 = (unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( (unsigned __int8 **)a2 != v15 )
    goto LABEL_14;
LABEL_30:
  sub_D5F1F0(*(_QWORD *)(a1 + 192), (__int64)v7);
  v24 = *(__int64 **)(a1 + 192);
  v25 = sub_BD5D20((__int64)v7);
  v26 = (__int64)v2[1];
  v98 = v25;
  v101 = 773;
  v99 = v27;
  v100[0] = ".sroa.phi";
  v28 = sub_D5C860(v24, v26, *((_DWORD *)v7 + 1) & 0x7FFFFFF, (__int64)&v98);
  v29 = *(_QWORD *)(a1 + 192);
  v30 = v28;
  v85 = (__int64)v2[9];
  v31 = *(_QWORD *)(sub_B43CB0((__int64)v2) + 80);
  if ( !v31 )
    BUG();
  v32 = *(_QWORD *)(v31 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v32 == v31 + 24 )
  {
    v34 = 0;
  }
  else
  {
    if ( !v32 )
      BUG();
    v33 = *(unsigned __int8 *)(v32 - 24);
    v34 = 0;
    v35 = v32 - 24;
    if ( (unsigned int)(v33 - 30) < 0xB )
      v34 = v35;
  }
  sub_D5F1F0(v29, v34);
  v93 = 0;
  v36 = *((_DWORD *)v7 + 1) & 0x7FFFFFF;
  v90 = 8LL * v36;
  if ( v36 )
  {
    v89 = (__int64)v2;
    while ( 1 )
    {
      v37 = *((_QWORD *)v7 - 1);
      v38 = *(_QWORD *)(v37 + 32LL * *((unsigned int *)v7 + 18) + v93);
      v39 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
      if ( !v39 )
        goto LABEL_93;
      v40 = *(_QWORD *)(v30 - 8);
      v41 = 0;
      v34 = *(unsigned int *)(v30 + 72);
      while ( v38 != *(_QWORD *)(v40 + 32 * v34 + 8 * v41) )
      {
        if ( v39 == (_DWORD)++v41 )
          goto LABEL_93;
      }
      if ( (int)v41 >= 0 )
      {
        v42 = *(_BYTE **)(v40 + 32LL * (int)v41);
        if ( (_DWORD)v34 != v39 )
          goto LABEL_43;
      }
      else
      {
LABEL_93:
        v87 = *(_QWORD *)(v37 + 4 * v93);
        v98 = (const char *)v100;
        v99 = 0x600000000LL;
        v60 = 4LL * (*(_DWORD *)(v89 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v89 + 7) & 0x40) != 0 )
        {
          v61 = *(unsigned __int8 ***)(v89 - 8);
          v62 = (__int64)&v61[v60];
        }
        else
        {
          v62 = v89;
          v61 = (unsigned __int8 **)(v89 - v60 * 8);
        }
        v63 = (unsigned __int8 **)v62;
        if ( (unsigned __int8 **)v62 == v61 )
        {
          v70 = (const char *)v100;
        }
        else
        {
          v86 = v38;
          v64 = v61;
          do
          {
            while ( 1 )
            {
              v65 = *v64;
              v66 = v7 == sub_BD3990(*v64, v34);
              v79 = (unsigned int)v99;
              v69 = (unsigned int)v99 + 1LL;
              if ( !v66 )
                break;
              if ( v69 > HIDWORD(v99) )
              {
                v34 = (__int64)v100;
                sub_C8D5F0((__int64)&v98, v100, v69, 8u, v67, v68);
                v79 = (unsigned int)v99;
              }
              v64 += 4;
              *(_QWORD *)&v98[8 * v79] = v87;
              LODWORD(v99) = v99 + 1;
              if ( v63 == v64 )
                goto LABEL_104;
            }
            if ( v69 > HIDWORD(v99) )
            {
              v34 = (__int64)v100;
              sub_C8D5F0((__int64)&v98, v100, v69, 8u, v67, v68);
              v79 = (unsigned int)v99;
            }
            v64 += 4;
            *(_QWORD *)&v98[8 * v79] = v65;
            LODWORD(v99) = v99 + 1;
          }
          while ( v63 != v64 );
LABEL_104:
          v38 = v86;
          v70 = v98;
        }
        v71 = *(_QWORD *)v70;
        if ( *(_BYTE *)v71 > 0x1Cu )
        {
          v72 = v84;
          LOWORD(v72) = 0;
          v84 = v72;
          sub_A88F30(*(_QWORD *)(a1 + 192), *(_QWORD *)(v71 + 40), *(_QWORD *)(v71 + 32), 0);
        }
        v73 = *(unsigned int ***)(a1 + 192);
        v74 = sub_B4DE20(v89);
        v94 = sub_BD5D20((__int64)v7);
        v95 = v75;
        v97 = 773;
        v96 = ".sroa.gep";
        v88 = sub_921130(v73, v85, *(_QWORD *)v98, (_BYTE **)v98 + 1, (unsigned int)v99 - 1LL, (__int64)&v94, v74);
        v76 = *(__int64 **)(a1 + 192);
        v94 = sub_BD5D20(v88);
        v96 = ".cast";
        v77 = *(_DWORD *)(v89 + 4);
        v97 = 773;
        v95 = v78;
        v42 = sub_291D720(v76, v88, *(_QWORD *)(*(_QWORD *)(v89 - 32LL * (v77 & 0x7FFFFFF)) + 8LL), (__int64)&v94);
        if ( v98 != (const char *)v100 )
          _libc_free((unsigned __int64)v98);
        v34 = *(unsigned int *)(v30 + 72);
        v39 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
        if ( (_DWORD)v34 != v39 )
          goto LABEL_43;
      }
      sub_B48D90(v30);
      v39 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
LABEL_43:
      v43 = (v39 + 1) & 0x7FFFFFF;
      *(_DWORD *)(v30 + 4) = v43 | *(_DWORD *)(v30 + 4) & 0xF8000000;
      v44 = *(_QWORD *)(v30 - 8) + 32LL * (unsigned int)(v43 - 1);
      if ( *(_QWORD *)v44 )
      {
        v45 = *(_QWORD *)(v44 + 8);
        **(_QWORD **)(v44 + 16) = v45;
        if ( v45 )
          *(_QWORD *)(v45 + 16) = *(_QWORD *)(v44 + 16);
      }
      *(_QWORD *)v44 = v42;
      if ( v42 )
      {
        v46 = *((_QWORD *)v42 + 2);
        *(_QWORD *)(v44 + 8) = v46;
        if ( v46 )
        {
          v34 = v44 + 8;
          *(_QWORD *)(v46 + 16) = v44 + 8;
        }
        *(_QWORD *)(v44 + 16) = v42 + 16;
        *((_QWORD *)v42 + 2) = v44;
      }
      v93 += 8;
      *(_QWORD *)(*(_QWORD *)(v30 - 8)
                + 32LL * *(unsigned int *)(v30 + 72)
                + 8LL * ((*(_DWORD *)(v30 + 4) & 0x7FFFFFFu) - 1)) = v38;
      if ( v90 == v93 )
      {
        v2 = (unsigned __int8 **)v89;
        break;
      }
    }
  }
  sub_2916060(a1, (__int64)v2, v7);
  sub_25DDDB0(a1 + 80, (__int64)v2);
  sub_BD84D0((__int64)v2, v30);
  sub_B43D60(v2);
  sub_AE6EC0(a1 + 80, v30);
  sub_2914720(a1, v30, v47, v48, v49, v50);
  return 1;
}
