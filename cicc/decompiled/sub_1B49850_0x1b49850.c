// Function: sub_1B49850
// Address: 0x1b49850
//
_BOOL8 __fastcall sub_1B49850(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *v10; // r12
  __int64 v11; // rbx
  unsigned __int64 v12; // r14
  __int64 v13; // rax
  int v15; // eax
  __int64 v16; // rax
  double v17; // xmm4_8
  double v18; // xmm5_8
  unsigned __int64 v19; // rax
  __int64 v20; // rcx
  int v21; // r8d
  int v22; // r9d
  unsigned __int8 *v23; // r15
  unsigned __int8 *v24; // rbx
  __int64 v25; // r13
  signed __int64 v26; // r13
  _QWORD *v27; // rbx
  _QWORD *v28; // rax
  __int64 v29; // r13
  __int64 v30; // rbx
  __int64 v31; // rsi
  _QWORD *v32; // r9
  unsigned __int64 v33; // r14
  __int64 v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  unsigned __int8 *v37; // rsi
  __int64 v38; // rax
  char v39; // al
  unsigned int v40; // eax
  __int64 v41; // rcx
  unsigned __int64 v42; // rdi
  __int64 v43; // rbx
  __int64 v44; // r15
  __int64 v45; // r12
  bool v46; // si
  __int64 v47; // r14
  unsigned __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  _QWORD *v52; // rdi
  __int64 v53; // rbx
  __int64 v54; // rax
  __int64 v55; // rdi
  char v56; // r15
  __int64 v57; // rax
  double v58; // xmm4_8
  double v59; // xmm5_8
  bool v60; // r15
  __int64 v61; // rcx
  _QWORD *v62; // rax
  __int64 v63; // rax
  __int64 v64; // r15
  _QWORD *v65; // rdi
  __int64 *v66; // r15
  __int64 v67; // r9
  unsigned int v68; // eax
  __int64 v69; // rcx
  __int64 *v70; // rbx
  __int64 *v71; // r15
  int v72; // r8d
  __int64 v73; // r9
  __int64 v74; // rax
  double v75; // xmm4_8
  double v76; // xmm5_8
  __int64 i; // r15
  double v78; // xmm4_8
  double v79; // xmm5_8
  unsigned __int8 *v80; // rdi
  __int64 *v81; // r15
  __int64 v82; // rdi
  __int64 v83; // [rsp+0h] [rbp-160h]
  __int64 v84; // [rsp+10h] [rbp-150h]
  char v86; // [rsp+37h] [rbp-129h]
  bool v87; // [rsp+38h] [rbp-128h]
  __int64 v88; // [rsp+38h] [rbp-128h]
  unsigned __int8 *v89; // [rsp+38h] [rbp-128h]
  __int64 v90[5]; // [rsp+40h] [rbp-120h] BYREF
  int v91; // [rsp+68h] [rbp-F8h]
  __int64 v92; // [rsp+70h] [rbp-F0h]
  __int64 v93; // [rsp+78h] [rbp-E8h]
  _BYTE *v94; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v95; // [rsp+98h] [rbp-C8h]
  _BYTE v96[64]; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned __int8 *v97; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v98; // [rsp+E8h] [rbp-78h]
  _BYTE v99[112]; // [rsp+F0h] [rbp-70h] BYREF

  v86 = *(_BYTE *)(*(_QWORD *)(a1 + 40) + 9LL);
  if ( !v86 )
    return 0;
  v10 = *(_QWORD **)(a2 + 40);
  v11 = a2 + 24;
  v87 = 0;
  if ( v10[6] == a2 + 24 )
    goto LABEL_8;
  v12 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v12 )
LABEL_18:
    BUG();
  while ( 1 )
  {
    if ( *(_BYTE *)(v12 - 8) == 78 )
    {
      v13 = *(_QWORD *)(v12 - 48);
      if ( *(_BYTE *)(v13 + 16) || (*(_BYTE *)(v13 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v13 + 36) - 35) > 3 )
        break;
    }
    if ( (unsigned __int8)sub_15F3040(v12 - 24) )
    {
      v15 = *(unsigned __int8 *)(v12 - 8);
      if ( (_BYTE)v15 == 55 )
        goto LABEL_13;
    }
    else
    {
      if ( !sub_15F3330(v12 - 24) )
        goto LABEL_14;
      v15 = *(unsigned __int8 *)(v12 - 8);
      if ( (_BYTE)v15 == 55 )
      {
LABEL_13:
        if ( (*(_BYTE *)(v12 - 6) & 1) != 0 )
          break;
        goto LABEL_14;
      }
    }
    if ( (_BYTE)v15 == 54 || (_BYTE)v15 == 59 || (_BYTE)v15 == 58 )
      goto LABEL_13;
    if ( (_BYTE)v15 == 74 )
    {
      v63 = sub_15E38F0(v10[7]);
      if ( (unsigned int)sub_14DD7D0(v63) != 10 )
        break;
    }
    else
    {
      v19 = (unsigned int)(v15 - 57);
      if ( (unsigned __int8)v19 > 0x1Fu )
        break;
      v20 = 2181038081LL;
      if ( !_bittest64(&v20, v19) )
        break;
    }
LABEL_14:
    if ( *(_QWORD *)(v12 - 16) )
    {
      v16 = sub_1599EF0(*(__int64 ***)(v12 - 24));
      sub_164D160(v12 - 24, v16, a3, a4, a5, a6, v17, v18, a9, a10);
    }
    sub_15F20C0((_QWORD *)(v12 - 24));
    v87 = v86;
    if ( v10[6] == v11 )
      goto LABEL_8;
    v12 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v12 )
      goto LABEL_18;
  }
  v11 = v10[6];
  if ( v11 )
  {
LABEL_8:
    if ( a2 != v11 - 24 )
      return v87;
    v97 = (unsigned __int8 *)v10[1];
    sub_15CDD40((__int64 *)&v97);
    v23 = v97;
    v94 = v96;
    v95 = 0x800000000LL;
    if ( v97 )
    {
      v24 = v97;
      v25 = 0;
      while ( 1 )
      {
        v24 = (unsigned __int8 *)*((_QWORD *)v24 + 1);
        if ( !v24 )
          break;
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700((__int64)v24) + 16) - 25) <= 9u )
        {
          v24 = (unsigned __int8 *)*((_QWORD *)v24 + 1);
          ++v25;
          if ( !v24 )
            goto LABEL_34;
        }
      }
LABEL_34:
      v26 = v25 + 1;
      if ( v26 > 8 )
      {
        sub_16CD150((__int64)&v94, v96, v26, 8, v21, v22);
        v27 = &v94[8 * (unsigned int)v95];
      }
      else
      {
        v27 = v96;
      }
      v28 = sub_1648700((__int64)v23);
LABEL_39:
      if ( v27 )
        *v27 = v28[5];
      while ( 1 )
      {
        v23 = (unsigned __int8 *)*((_QWORD *)v23 + 1);
        if ( !v23 )
          break;
        v28 = sub_1648700((__int64)v23);
        if ( (unsigned __int8)(*((_BYTE *)v28 + 16) - 25) <= 9u )
        {
          ++v27;
          goto LABEL_39;
        }
      }
      v29 = (unsigned int)(v95 + v26);
      LODWORD(v95) = v29;
      if ( (_DWORD)v29 )
      {
        v30 = 0;
        while ( 1 )
        {
          v33 = sub_157EBA0(*(_QWORD *)&v94[v30]);
          v34 = sub_16498A0(v33);
          v92 = 0;
          v93 = 0;
          v37 = *(unsigned __int8 **)(v33 + 48);
          v90[3] = v34;
          v91 = 0;
          v38 = *(_QWORD *)(v33 + 40);
          v90[0] = 0;
          v90[1] = v38;
          v90[4] = 0;
          v90[2] = v33 + 24;
          v97 = v37;
          if ( v37 )
          {
            sub_1623A60((__int64)&v97, (__int64)v37, 2);
            if ( v90[0] )
              sub_161E7C0((__int64)v90, v90[0]);
            v90[0] = (__int64)v97;
            if ( v97 )
              sub_1623210((__int64)&v97, v97, (__int64)v90);
          }
          v39 = *(_BYTE *)(v33 + 16);
          if ( v39 == 26 )
          {
            v31 = *(_QWORD *)(v33 - 24);
            if ( (*(_DWORD *)(v33 + 20) & 0xFFFFFFF) == 1 )
            {
              if ( v31 != 0 && v10 == (_QWORD *)v31 )
              {
                v88 = sub_16498A0(v33);
                v52 = sub_1648A60(56, 0);
                if ( v52 )
                  sub_15F82A0((__int64)v52, v88, v33);
                sub_15F20C0((_QWORD *)v33);
                v87 = v31 != 0 && v10 == (_QWORD *)v31;
              }
            }
            else
            {
              v32 = *(_QWORD **)(v33 - 48);
              if ( v31 && v10 == (_QWORD *)v31 )
              {
                sub_1B44660(v90, *(_QWORD *)(v33 - 48));
                sub_1B44FE0(v33);
              }
              else if ( v32 != 0 && v10 == v32 )
              {
                v87 = v32 != 0 && v10 == v32;
                sub_1B44660(v90, v31);
                sub_1B44FE0(v33);
              }
            }
            goto LABEL_49;
          }
          if ( v39 == 27 )
          {
            v40 = (*(_DWORD *)(v33 + 20) & 0xFFFFFFFu) >> 1;
            v41 = v40 - 1;
            if ( v40 != 1 )
            {
              v84 = v30;
              v42 = v33;
              v43 = (__int64)v10;
              v44 = v33;
              v45 = v33;
              v46 = v87;
              v47 = 0;
              while ( 1 )
              {
                v50 = 24;
                if ( (_DWORD)v47 != -2 )
                  v50 = 24LL * (unsigned int)(2 * v47 + 3);
                if ( (*(_BYTE *)(v44 + 23) & 0x40) != 0 )
                  v48 = *(_QWORD *)(v44 - 8);
                else
                  v48 = v42 - 24LL * (*(_DWORD *)(v44 + 20) & 0xFFFFFFF);
                v49 = *(_QWORD *)(v48 + v50);
                if ( v49 && v43 == v49 )
                {
                  sub_157F2D0(v43, *(_QWORD *)(v45 + 40), 0);
                  v46 = v86;
                  v44 = sub_15FFDB0(v45, v44, v47);
                  v47 = v51;
                  v41 = ((*(_DWORD *)(v45 + 20) & 0xFFFFFFFu) >> 1) - 1;
                  if ( v51 == v41 )
                  {
LABEL_71:
                    v10 = (_QWORD *)v43;
                    v87 = v46;
                    v30 = v84;
                    goto LABEL_49;
                  }
                }
                else if ( ++v47 == v41 )
                {
                  goto LABEL_71;
                }
                v42 = v44;
              }
            }
            goto LABEL_49;
          }
          if ( v39 == 29 )
            break;
          if ( v39 != 34 )
          {
            if ( v39 != 32 )
              goto LABEL_49;
LABEL_99:
            v64 = sub_16498A0(v33);
            v65 = sub_1648A60(56, 0);
            if ( v65 )
              sub_15F82A0((__int64)v65, v64, v33);
            sub_15F20C0((_QWORD *)v33);
            v87 = v86;
LABEL_49:
            if ( v90[0] )
              sub_161E7C0((__int64)v90, v90[0]);
            goto LABEL_51;
          }
          v56 = *(_BYTE *)(v33 + 18);
          v57 = sub_13CF970(v33);
          v60 = v56 & 1;
          v61 = v57;
          if ( !v60 )
          {
            v66 = (__int64 *)(v57 + 24);
            v73 = sub_13CF970(v33);
            v68 = *(_DWORD *)(v33 + 20) & 0xFFFFFFF;
            v69 = v73 + 24LL * v68;
            if ( (__int64 *)v69 != v66 )
              goto LABEL_103;
            goto LABEL_112;
          }
          v62 = *(_QWORD **)(v57 + 24);
          if ( !v62 || v10 != v62 )
          {
            v66 = (__int64 *)(v61 + 48);
            v67 = sub_13CF970(v33);
            v68 = *(_DWORD *)(v33 + 20) & 0xFFFFFFF;
            v69 = v67 + 24LL * v68;
            if ( (__int64 *)v69 == v66 )
              goto LABEL_109;
LABEL_103:
            v83 = v30;
            v70 = v66;
            v71 = (__int64 *)v69;
            do
            {
              if ( v10 == (_QWORD *)sub_15A5110(*v70) )
              {
                v71 -= 3;
                sub_15F7E90(v33, v70);
                v87 = v86;
              }
              else
              {
                v70 += 3;
              }
            }
            while ( v71 != v70 );
            v30 = v83;
            v68 = *(_DWORD *)(v33 + 20) & 0xFFFFFFF;
            if ( (*(_BYTE *)(v33 + 18) & 1) != 0 )
            {
LABEL_109:
              if ( v68 != 2 )
                goto LABEL_49;
              v74 = sub_13CF970(v33);
              sub_164D160(*(_QWORD *)(v33 + 40), *(_QWORD *)(v74 + 24), a3, a4, a5, a6, v75, v76, a9, a10);
              goto LABEL_99;
            }
LABEL_112:
            if ( v68 != 1 )
              goto LABEL_49;
            for ( i = *(_QWORD *)(*(_QWORD *)(v33 + 40) + 8LL); i; i = *(_QWORD *)(i + 8) )
            {
              if ( (unsigned __int8)(*((_BYTE *)sub_1648700(i) + 16) - 25) <= 9u )
                break;
            }
            v97 = v99;
            v98 = 0x800000000LL;
            sub_1B49760((__int64)&v97, i, 0, v69, v72, v73);
            v80 = v97;
            v81 = (__int64 *)v97;
            v89 = &v97[8 * (unsigned int)v98];
            if ( v97 != v89 )
            {
              do
              {
                v82 = *v81++;
                sub_1AF0970(v82, 0, a3, a4, a5, a6, v78, v79, a9, a10);
              }
              while ( v89 != (unsigned __int8 *)v81 );
              v80 = v97;
            }
            if ( v80 != v99 )
              _libc_free((unsigned __int64)v80);
            goto LABEL_99;
          }
          sub_1AF0970(*(_QWORD *)(v33 + 40), 0, a3, a4, a5, a6, v58, v59, a9, a10);
          sub_17CD270(v90);
          v87 = v60;
LABEL_51:
          v30 += 8;
          if ( 8 * v29 == v30 )
            goto LABEL_80;
        }
        if ( v10 == *(_QWORD **)(v33 - 24) )
        {
          sub_1AF0970(*(_QWORD *)(v33 + 40), 0, a3, a4, a5, a6, v35, v36, a9, a10);
          v87 = v86;
        }
        goto LABEL_49;
      }
    }
    else
    {
      LODWORD(v95) = 0;
    }
LABEL_80:
    v53 = v10[1];
    if ( v53 )
    {
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v53) + 16) - 25) > 9u )
      {
        v53 = *(_QWORD *)(v53 + 8);
        if ( !v53 )
          goto LABEL_85;
      }
    }
    else
    {
LABEL_85:
      v54 = *(_QWORD *)(v10[7] + 80LL);
      if ( !v54 || v10 != (_QWORD *)(v54 - 24) )
      {
        sub_157F980((__int64)v10);
        v55 = *(_QWORD *)(a1 + 32);
        v87 = v86;
        if ( v55 )
          sub_1B44390(v55, (__int64)v10);
      }
    }
    if ( v94 != v96 )
      _libc_free((unsigned __int64)v94);
  }
  return v87;
}
