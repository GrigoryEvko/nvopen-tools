// Function: sub_2D63A40
// Address: 0x2d63a40
//
__int64 __fastcall sub_2D63A40(__int64 a1, __int64 a2, _DWORD *a3)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // r8
  unsigned int v6; // eax
  unsigned __int8 *v7; // r15
  unsigned __int8 v8; // al
  unsigned __int8 *v9; // r10
  unsigned __int8 *v10; // r13
  unsigned __int8 *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned __int8 *v14; // r10
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 i; // r13
  char v18; // dl
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned int v21; // r13d
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // r13
  __int64 v28; // r15
  unsigned __int8 *v29; // r12
  __int64 v30; // rax
  __int64 v31; // rbx
  unsigned __int8 *v32; // rax
  __int64 v33; // rdi
  __int64 (*v34)(); // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 *v39; // r14
  int v40; // r15d
  _BYTE *v41; // r13
  __int64 v42; // rbx
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  __int64 *v45; // rbx
  unsigned __int64 v46; // r15
  unsigned __int64 v47; // rax
  __int64 v48; // rdx
  bool v49; // cf
  unsigned __int64 v50; // rax
  _BYTE *v51; // rax
  _QWORD *v52; // r14
  _QWORD *v53; // rcx
  __int64 v54; // r12
  _BYTE **v55; // rbx
  _BYTE **v56; // r13
  __int64 v57; // r15
  _BYTE *v58; // rdi
  _QWORD *v59; // rax
  _DWORD *v60; // rax
  __int64 v61; // rdx
  _DWORD *v62; // rsi
  __int64 v63; // rdi
  __int64 v64; // rdx
  _DWORD *v65; // rdx
  __int64 v66; // rdi
  __int64 (*v67)(); // rax
  __int64 v68; // r14
  __int64 v69; // rdx
  __int64 *v70; // r13
  __int64 v71; // rdi
  unsigned __int64 v72; // rax
  __int64 v73; // r12
  __int64 v74; // rdi
  __int64 (*v75)(); // rax
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  unsigned __int8 *v84; // rax
  __int64 v85; // [rsp+8h] [rbp-188h]
  unsigned __int64 v86; // [rsp+30h] [rbp-160h]
  __int64 v87; // [rsp+30h] [rbp-160h]
  __int64 v88; // [rsp+38h] [rbp-158h]
  __int64 v89; // [rsp+38h] [rbp-158h]
  unsigned __int64 v90; // [rsp+38h] [rbp-158h]
  __int64 v91; // [rsp+40h] [rbp-150h]
  __int64 v92; // [rsp+40h] [rbp-150h]
  unsigned __int8 *v93; // [rsp+40h] [rbp-150h]
  unsigned __int8 *v94; // [rsp+48h] [rbp-148h]
  __int64 v95; // [rsp+48h] [rbp-148h]
  __int64 *v96; // [rsp+48h] [rbp-148h]
  _QWORD *v97; // [rsp+48h] [rbp-148h]
  unsigned __int64 v99; // [rsp+50h] [rbp-140h]
  unsigned __int8 v101; // [rsp+58h] [rbp-138h]
  _BYTE v102[48]; // [rsp+60h] [rbp-130h] BYREF
  _BYTE *v103; // [rsp+90h] [rbp-100h] BYREF
  __int64 v104; // [rsp+98h] [rbp-F8h]
  _BYTE v105[32]; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 *v106; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v107; // [rsp+C8h] [rbp-C8h]
  _BYTE v108[32]; // [rsp+D0h] [rbp-C0h] BYREF
  _BYTE *v109; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v110; // [rsp+F8h] [rbp-98h]
  _BYTE v111[32]; // [rsp+100h] [rbp-90h] BYREF
  __int64 v112; // [rsp+120h] [rbp-70h] BYREF
  char *v113; // [rsp+128h] [rbp-68h]
  __int64 v114; // [rsp+130h] [rbp-60h]
  int v115; // [rsp+138h] [rbp-58h]
  char v116; // [rsp+13Ch] [rbp-54h]
  char v117; // [rsp+140h] [rbp-50h] BYREF

  v3 = a2;
  v4 = sub_986580(a2);
  if ( !v4 )
    return 0;
  v5 = v4;
  if ( *(_BYTE *)v4 != 30 )
    return 0;
  v6 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
  if ( v6 )
  {
    v7 = *(unsigned __int8 **)(v5 - 32LL * v6);
    if ( v7 )
    {
      v8 = *v7;
      v9 = 0;
      if ( *v7 == 78 )
      {
        v9 = v7;
        v8 = **((_BYTE **)v7 - 4);
        v7 = (unsigned __int8 *)*((_QWORD *)v7 - 4);
      }
      v10 = 0;
      if ( v8 == 93 )
      {
        v60 = (_DWORD *)*((_QWORD *)v7 + 9);
        v61 = 4LL * *((unsigned int *)v7 + 20);
        v62 = &v60[(unsigned __int64)v61 / 4];
        v63 = v61 >> 2;
        v64 = v61 >> 4;
        if ( !v64 )
          goto LABEL_150;
        v65 = &v60[4 * v64];
        do
        {
          if ( *v60 )
            goto LABEL_119;
          if ( v60[1] )
          {
            ++v60;
            goto LABEL_119;
          }
          if ( v60[2] )
          {
            v60 += 2;
            goto LABEL_119;
          }
          if ( v60[3] )
          {
            v60 += 3;
            goto LABEL_119;
          }
          v60 += 4;
        }
        while ( v65 != v60 );
        v63 = v62 - v60;
LABEL_150:
        if ( v63 == 2 )
          goto LABEL_158;
        if ( v63 != 3 )
        {
          if ( v63 == 1 )
            goto LABEL_153;
          goto LABEL_120;
        }
        if ( *v60 )
          goto LABEL_119;
        ++v60;
LABEL_158:
        if ( *v60 )
          goto LABEL_119;
        ++v60;
LABEL_153:
        if ( *v60 )
        {
LABEL_119:
          if ( v62 == v60 )
            goto LABEL_120;
          return 0;
        }
LABEL_120:
        v10 = v7;
        v8 = **((_BYTE **)v7 - 4);
        v7 = (unsigned __int8 *)*((_QWORD *)v7 - 4);
      }
      v11 = 0;
      if ( v8 == 84 )
      {
        v11 = v7;
        if ( v3 != *((_QWORD *)v7 + 5) )
          return 0;
      }
    }
    else
    {
      v9 = 0;
      v10 = 0;
      v11 = 0;
    }
  }
  else
  {
    v7 = 0;
    v9 = 0;
    v10 = 0;
    v11 = 0;
  }
  v91 = v5;
  v103 = v105;
  v94 = v9;
  v104 = 0x400000000LL;
  v12 = sub_AA4FF0(v3);
  v14 = v94;
  v15 = (__int64)v10;
  v16 = v91;
  for ( i = v12; ; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      if ( !i )
        BUG();
      v18 = *(_BYTE *)(i - 24);
      v19 = i - 24;
      if ( v18 != 85 )
        break;
      v13 = *(_QWORD *)(i - 56);
      if ( v13 )
      {
        if ( !*(_BYTE *)v13
          && *(_QWORD *)(v13 + 24) == *(_QWORD *)(i + 56)
          && (*(_BYTE *)(v13 + 33) & 0x20) != 0
          && (unsigned int)(*(_DWORD *)(v13 + 36) - 68) <= 3
          || v19 == v15
          || (unsigned __int8 *)v19 == v14 )
        {
          goto LABEL_24;
        }
        if ( !*(_BYTE *)v13
          && *(_QWORD *)(v13 + 24) == *(_QWORD *)(i + 56)
          && (*(_BYTE *)(v13 + 33) & 0x20) != 0
          && *(_DWORD *)(v13 + 36) == 291 )
        {
          i = *(_QWORD *)(i + 8);
        }
        else
        {
LABEL_32:
          v23 = *(_QWORD *)(i - 56);
          if ( !v13 )
            goto LABEL_16;
          if ( !*(_BYTE *)v13 )
          {
            v13 = i - 24;
            if ( *(_QWORD *)(v23 + 24) != *(_QWORD *)(v19 + 80) )
              goto LABEL_43;
            goto LABEL_35;
          }
LABEL_45:
          if ( !v13
            || *(_BYTE *)v13
            || *(_QWORD *)(v13 + 24) != *(_QWORD *)(i + 56)
            || (*(_BYTE *)(v13 + 33) & 0x20) == 0
            || *(_DWORD *)(v13 + 36) != 171 )
          {
            goto LABEL_16;
          }
          if ( **(_BYTE **)(i - 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF) - 24) == 84 )
          {
LABEL_24:
            i = *(_QWORD *)(i + 8);
          }
          else
          {
            v24 = (unsigned int)v104;
            v13 = HIDWORD(v104);
            v25 = (unsigned int)v104 + 1LL;
            if ( v25 > HIDWORD(v104) )
            {
              v87 = v15;
              v89 = v16;
              v93 = v14;
              sub_C8D5F0((__int64)&v103, v105, v25, 8u, v16, v15);
              v24 = (unsigned int)v104;
              v15 = v87;
              v16 = v89;
              v14 = v93;
            }
            *(_QWORD *)&v103[8 * v24] = v19;
            LODWORD(v104) = v104 + 1;
            i = *(_QWORD *)(i + 8);
          }
        }
      }
      else
      {
        if ( v19 == v15 )
          goto LABEL_24;
        if ( (unsigned __int8 *)v19 != v14 )
          goto LABEL_32;
        i = *(_QWORD *)(i + 8);
      }
    }
    if ( v19 == v15 || (unsigned __int8 *)v19 == v14 )
      goto LABEL_24;
    if ( v18 != 78 )
      break;
    v20 = *(_QWORD *)(i - 8);
    if ( !v20 )
      break;
    if ( *(_QWORD *)(v20 + 8) )
      break;
    v13 = *(_QWORD *)(v20 + 24);
    if ( *(_BYTE *)v13 != 85 )
      break;
    v23 = *(_QWORD *)(v13 - 32);
    if ( !v23 || *(_BYTE *)v23 )
      break;
    if ( *(_QWORD *)(v23 + 24) != *(_QWORD *)(v13 + 80) )
      goto LABEL_43;
LABEL_35:
    if ( (*(_BYTE *)(v23 + 33) & 0x20) == 0 || *(_DWORD *)(v23 + 36) != 210 )
    {
LABEL_43:
      if ( v18 != 85 )
        break;
      v13 = *(_QWORD *)(i - 56);
      goto LABEL_45;
    }
  }
LABEL_16:
  v21 = 0;
  if ( v16 != v19 )
    goto LABEL_17;
  v26 = (__int64)v111;
  v95 = *(_QWORD *)(v3 + 72);
  v106 = (__int64 *)v108;
  v107 = 0x400000000LL;
  v109 = v111;
  v110 = 0x400000000LL;
  if ( v11 )
  {
    if ( (*((_DWORD *)v11 + 1) & 0x7FFFFFF) != 0 )
    {
      v88 = v19;
      v27 = 0;
      v86 = v3;
      v92 = 8LL * (*((_DWORD *)v11 + 1) & 0x7FFFFFF);
      while ( 1 )
      {
        v29 = sub_BD3990(*(unsigned __int8 **)(*((_QWORD *)v11 - 1) + 4 * v27), v26);
        if ( *v29 <= 0x1Cu )
          break;
        v28 = *(_QWORD *)(*((_QWORD *)v11 - 1) + 32LL * *((unsigned int *)v11 + 18) + v27);
        if ( *v29 != 85 )
          goto LABEL_64;
        v30 = *((_QWORD *)v29 + 2);
        if ( v30 )
        {
          if ( *(_QWORD *)(v30 + 8)
            || *((_QWORD *)v29 + 5) != v28
            || (v66 = *(_QWORD *)(a1 + 16), v67 = *(__int64 (**)())(*(_QWORD *)v66 + 2344LL), v67 == sub_2D566D0) )
          {
            if ( !v28 )
              goto LABEL_66;
          }
          else
          {
            v26 = (__int64)v29;
            if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int8 *))v67)(v66, v29) )
            {
              v26 = (__int64)v29;
              if ( (unsigned __int8)sub_34B9300(v95, v29, v88, *(_QWORD *)(a1 + 16), 0) )
              {
                sub_B1A4E0((__int64)&v106, v28);
                v26 = (__int64)v29;
                sub_11EECC0((__int64)&v109, (__int64)v29, v80, v81, v82, v83);
                goto LABEL_66;
              }
            }
            v31 = (__int64)v29;
            if ( !v28 )
              goto LABEL_72;
          }
        }
        else
        {
          v31 = (__int64)v29;
          if ( !v28 )
            goto LABEL_73;
        }
        v31 = (__int64)v29;
        if ( v86 == sub_AA56F0(v28) )
          goto LABEL_101;
LABEL_72:
        if ( !*(_QWORD *)(v31 + 16) )
        {
LABEL_73:
          v26 = v31;
          if ( sub_2D58330(*(__int64 **)(a1 + 48), v31) )
          {
            v32 = *(unsigned __int8 **)(v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF));
            if ( v29 == v32 )
            {
              if ( v32 )
              {
                v33 = *(_QWORD *)(a1 + 16);
                v34 = *(__int64 (**)())(*(_QWORD *)v33 + 2344LL);
                if ( v34 != sub_2D566D0 )
                {
                  v26 = v31;
                  if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))v34)(v33, v31) )
                  {
                    v26 = v31;
                    if ( (unsigned __int8)sub_34B9300(v95, v31, v88, *(_QWORD *)(a1 + 16), 0) )
                    {
                      sub_B1A4E0((__int64)&v106, v28);
                      v26 = v31;
                      sub_11EECC0((__int64)&v109, v31, v35, v36, v37, v38);
                    }
                  }
                }
              }
            }
          }
        }
LABEL_66:
        v27 += 8;
        if ( v92 == v27 )
        {
          v19 = v88;
          v3 = v86;
          goto LABEL_85;
        }
      }
      v28 = *(_QWORD *)(*((_QWORD *)v11 - 1) + 32LL * *((unsigned int *)v11 + 18) + v27);
LABEL_64:
      if ( !v28 || v86 != sub_AA56F0(v28) )
        goto LABEL_66;
LABEL_101:
      v50 = sub_986580(v28);
      v26 = 1;
      v51 = (_BYTE *)sub_B46BC0(v50, 1);
      v31 = (__int64)v51;
      if ( !v51 || *v51 != 85 )
        goto LABEL_66;
      goto LABEL_72;
    }
  }
  else
  {
    v68 = *(_QWORD *)(v3 + 16);
    v112 = 0;
    v113 = &v117;
    v114 = 4;
    v115 = 0;
    v116 = 1;
    if ( v68 )
    {
      while ( 1 )
      {
        v69 = *(_QWORD *)(v68 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v69 - 30) <= 0xAu )
          break;
        v68 = *(_QWORD *)(v68 + 8);
        if ( !v68 )
          goto LABEL_85;
      }
      v90 = v3;
LABEL_132:
      v70 = *(__int64 **)(v69 + 40);
      sub_D695C0((__int64)v102, (__int64)&v112, v70, v13, v16, v15);
      if ( v102[32] )
      {
        v71 = (v70[6] & 0xFFFFFFFFFFFFFFF8LL) - 24;
        if ( (v70[6] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          v71 = 0;
        v72 = sub_B46BC0(v71, 1);
        v73 = v72;
        if ( v72 )
        {
          if ( *(_BYTE *)v72 == 85 && !*(_QWORD *)(v72 + 16) )
          {
            v74 = *(_QWORD *)(a1 + 16);
            v75 = *(__int64 (**)())(*(_QWORD *)v74 + 2344LL);
            if ( v75 != sub_2D566D0 )
            {
              if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))v75)(v74, v73) )
              {
                if ( (unsigned __int8)sub_34B9300(v95, v73, v19, *(_QWORD *)(a1 + 16), 0) )
                {
                  if ( !v7
                    || (unsigned int)*v7 - 12 <= 1
                    || sub_2D58330(*(__int64 **)(a1 + 48), v73)
                    && (v84 = *(unsigned __int8 **)(v73 - 32LL * (*(_DWORD *)(v73 + 4) & 0x7FFFFFF)), v84 == v7)
                    && v84 )
                  {
                    sub_B1A4E0((__int64)&v106, (__int64)v70);
                    sub_11EECC0((__int64)&v109, v73, v76, v77, v78, v79);
                  }
                }
              }
            }
          }
        }
      }
      while ( 1 )
      {
        v68 = *(_QWORD *)(v68 + 8);
        if ( !v68 )
          break;
        v69 = *(_QWORD *)(v68 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v69 - 30) <= 0xAu )
          goto LABEL_132;
      }
      v3 = v90;
      if ( !v116 )
        _libc_free((unsigned __int64)v113);
    }
LABEL_85:
    v21 = 0;
    v96 = &v106[(unsigned int)v107];
    if ( v96 != v106 )
    {
      v39 = v106;
      v40 = 0;
      v41 = (_BYTE *)v19;
      do
      {
        v42 = *v39;
        v43 = sub_986580(*v39);
        if ( *(_BYTE *)v43 == 31 && (*(_DWORD *)(v43 + 4) & 0x7FFFFFF) == 1 )
        {
          v44 = *(_QWORD *)(v43 - 32);
          if ( v3 == v44 )
          {
            if ( v44 )
            {
              sub_F355A0(v41, v3, v42, 0);
              v45 = *(__int64 **)(a1 + 64);
              v46 = sub_FDD860(v45, *v39);
              v47 = sub_FDD860(*(__int64 **)(a1 + 64), v3);
              v48 = v47 - v46;
              v49 = v46 < v47;
              v40 = 1;
              if ( !v49 )
                v48 = 0;
              sub_FE1040(v45, v3, v48);
              *a3 = 1;
            }
          }
        }
        ++v39;
      }
      while ( v96 != v39 );
      v21 = v40;
      if ( (_BYTE)v40 )
      {
        if ( (*(_WORD *)(v3 + 2) & 0x7FFF) == 0 )
        {
          v112 = *(_QWORD *)(v3 + 16);
          sub_2D63A10(&v112);
          if ( !v112 )
          {
            v52 = v109;
            v53 = &v109[8 * (unsigned int)v110];
            if ( v53 != (_QWORD *)v109 )
            {
              v101 = v40;
              v99 = v3;
              v54 = v85;
              do
              {
                v55 = (_BYTE **)v103;
                v56 = (_BYTE **)&v103[8 * (unsigned int)v104];
                v57 = *v52 + 24LL;
                if ( v56 != (_BYTE **)v103 )
                {
                  do
                  {
                    v58 = *v55;
                    v97 = v53;
                    LOWORD(v54) = 0;
                    ++v55;
                    v59 = (_QWORD *)sub_B47F80(v58);
                    sub_B44220(v59, v57, v54);
                    v53 = v97;
                  }
                  while ( v56 != v55 );
                }
                ++v52;
              }
              while ( v53 != v52 );
              v21 = v101;
              v3 = v99;
            }
            sub_AA5450((_QWORD *)v3);
          }
        }
      }
    }
    if ( v109 != v111 )
      _libc_free((unsigned __int64)v109);
  }
  if ( v106 != (__int64 *)v108 )
    _libc_free((unsigned __int64)v106);
LABEL_17:
  if ( v103 != v105 )
    _libc_free((unsigned __int64)v103);
  return v21;
}
