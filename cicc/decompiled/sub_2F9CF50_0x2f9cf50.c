// Function: sub_2F9CF50
// Address: 0x2f9cf50
//
__int64 __fastcall sub_2F9CF50(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  void *v4; // rax
  __int64 v5; // rdx
  _BYTE *v6; // rdi
  __int64 v7; // rax
  __int8 *v8; // rsi
  size_t v9; // rdx
  unsigned __int64 *v10; // r13
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rdi
  unsigned __int64 *v13; // rbx
  unsigned __int64 *v14; // r12
  unsigned __int64 v15; // rdi
  char v17; // al
  __int8 *v18; // rsi
  size_t v19; // rdx
  __int64 *v20; // rax
  unsigned __int64 *v21; // r12
  unsigned __int64 *v22; // r14
  unsigned __int64 v23; // rdi
  __int64 v24; // r12
  __int64 v25; // rax
  int v26; // ecx
  __int64 v27; // rsi
  int v28; // ecx
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rdi
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 *v37; // r15
  __int64 v38; // r14
  __int64 v39; // r12
  int v40; // esi
  char v41; // cl
  int v42; // eax
  _BYTE *v43; // r13
  __int64 v44; // rax
  int v45; // r13d
  signed __int64 v46; // rbx
  __int64 **v47; // r15
  __int64 v48; // r9
  __int64 v49; // r8
  __int64 v50; // r11
  __int64 v51; // r8
  __int64 v52; // r8
  unsigned __int64 v53; // rdx
  __int64 v54; // rax
  unsigned __int8 **v55; // rdi
  int v56; // ecx
  unsigned __int8 **v57; // r10
  unsigned __int64 v58; // rcx
  int v59; // edx
  __int64 v60; // r15
  bool v61; // of
  __int64 v62; // rax
  signed __int64 v63; // rax
  __int64 v64; // rdx
  int v65; // eax
  int v66; // r8d
  unsigned __int64 v67; // [rsp+8h] [rbp-608h]
  unsigned __int64 v68; // [rsp+10h] [rbp-600h]
  __int64 *v69; // [rsp+20h] [rbp-5F0h]
  __int64 *v71; // [rsp+40h] [rbp-5D0h]
  unsigned __int8 v72; // [rsp+4Fh] [rbp-5C1h]
  unsigned __int8 **v73; // [rsp+50h] [rbp-5C0h]
  __int64 v74; // [rsp+60h] [rbp-5B0h]
  __int64 v75; // [rsp+68h] [rbp-5A8h]
  __int64 v76; // [rsp+70h] [rbp-5A0h]
  __int64 v77; // [rsp+78h] [rbp-598h]
  signed __int64 v78; // [rsp+80h] [rbp-590h]
  int v79; // [rsp+88h] [rbp-588h]
  __int64 v80; // [rsp+88h] [rbp-588h]
  unsigned __int64 v81; // [rsp+90h] [rbp-580h] BYREF
  unsigned __int64 v82; // [rsp+98h] [rbp-578h] BYREF
  unsigned __int8 **v83; // [rsp+A0h] [rbp-570h] BYREF
  __int64 v84; // [rsp+A8h] [rbp-568h]
  _BYTE v85[32]; // [rsp+B0h] [rbp-560h] BYREF
  _QWORD v86[10]; // [rsp+D0h] [rbp-540h] BYREF
  unsigned __int64 *v87; // [rsp+120h] [rbp-4F0h]
  unsigned int v88; // [rsp+128h] [rbp-4E8h]
  char v89; // [rsp+130h] [rbp-4E0h] BYREF
  _QWORD v90[10]; // [rsp+280h] [rbp-390h] BYREF
  unsigned __int64 *v91; // [rsp+2D0h] [rbp-340h]
  unsigned int v92; // [rsp+2D8h] [rbp-338h]
  char v93; // [rsp+2E0h] [rbp-330h] BYREF
  void *v94; // [rsp+430h] [rbp-1E0h] BYREF
  __int64 v95; // [rsp+438h] [rbp-1D8h]
  __int64 v96; // [rsp+440h] [rbp-1D0h]
  __int64 v97; // [rsp+448h] [rbp-1C8h]
  __int64 v98; // [rsp+450h] [rbp-1C0h]
  __int64 v99; // [rsp+458h] [rbp-1B8h]
  unsigned __int64 v100; // [rsp+460h] [rbp-1B0h]
  __int64 v101; // [rsp+468h] [rbp-1A8h]
  __int64 v102; // [rsp+470h] [rbp-1A0h]
  __int64 *v103; // [rsp+478h] [rbp-198h]
  unsigned __int64 *v104; // [rsp+480h] [rbp-190h]
  unsigned int v105; // [rsp+488h] [rbp-188h]
  char v106; // [rsp+490h] [rbp-180h] BYREF

  v2 = a1;
  v71 = *(__int64 **)(a2 + 8);
  sub_B174A0((__int64)v86, (__int64)"select-optimize", (__int64)"SelectOpti", 10, *v71);
  sub_B176B0((__int64)v90, (__int64)"select-optimize", (__int64)"SelectOpti", 10, *v71);
  v3 = *(_QWORD *)(a1 + 48);
  v4 = (void *)sub_FDD2C0(*(__int64 **)(a1 + 40), *(_QWORD *)(*v71 + 40), 0);
  v95 = v5;
  v94 = v4;
  if ( (_BYTE)v5 )
  {
    v17 = sub_D84450(v3, (unsigned __int64)v94);
    v8 = "Not converted to branch because of cold basic block. ";
    v9 = 53;
    if ( v17 )
      goto LABEL_4;
    v6 = (_BYTE *)*v71;
    if ( (*(_BYTE *)(*v71 + 7) & 0x20) != 0 )
    {
LABEL_3:
      v7 = sub_B91C10((__int64)v6, 15);
      v8 = "Not converted to branch because of unpredictable branch. ";
      v9 = 57;
      if ( v7 )
      {
LABEL_4:
        sub_B18290((__int64)v90, v8, v9);
        sub_1049740(*(__int64 **)(v2 + 56), (__int64)v90);
        v72 = 0;
        goto LABEL_5;
      }
      v6 = (_BYTE *)*v71;
    }
  }
  else
  {
    v6 = (_BYTE *)*v71;
    if ( (*(_BYTE *)(*v71 + 7) & 0x20) != 0 )
      goto LABEL_3;
  }
  if ( (unsigned __int8)sub_2F9A6D0(v2, v6) )
  {
    v18 = "Converted to branch because of highly predictable branch. ";
    v19 = 58;
    v72 = *(_BYTE *)(*(_QWORD *)(v2 + 16) + 537004LL);
    if ( v72 )
    {
LABEL_82:
      sub_B18290((__int64)v86, v18, v19);
      sub_1049740(*(__int64 **)(v2 + 56), (__int64)v86);
      goto LABEL_5;
    }
  }
  v20 = *(__int64 **)(a2 + 8);
  if ( *(_BYTE *)*v20 == 86 && (v72 = sub_BC8C50(*v20, &v81, &v82)) != 0 )
  {
    v33 = v82;
    v34 = v81;
    v78 = v82;
    v67 = v82 + v81;
    if ( v82 > v81 )
      v33 = v81;
    if ( (v82 + v81) * (unsigned int)qword_5025BC8 > 100 * v33 )
    {
      v35 = *(_QWORD *)(a2 + 8);
      v36 = 16LL * *(unsigned int *)(a2 + 16);
      v73 = (unsigned __int8 **)(v35 + v36);
      if ( v35 != v35 + v36 )
      {
        v37 = *(__int64 **)(a2 + 8);
        v38 = v2;
        v68 = v67 >> 1;
        while ( 1 )
        {
          v39 = *v37;
          v40 = *((_DWORD *)v37 + 3);
          v41 = *((_BYTE *)v37 + 8);
          v42 = *(unsigned __int8 *)*v37;
          if ( v34 < v78 )
          {
            if ( v41 )
            {
              if ( (_BYTE)v42 == 86 )
              {
                v43 = *(_BYTE **)(v39 - 32);
                goto LABEL_56;
              }
              if ( (unsigned int)(v42 - 42) > 0x11 )
LABEL_128:
                BUG();
              v43 = *(_BYTE **)(v39 + 32LL * (unsigned int)(1 - v40) - 64);
              if ( !v43 )
                goto LABEL_90;
            }
            else
            {
              if ( (_BYTE)v42 != 86 )
              {
LABEL_104:
                if ( (unsigned int)(v42 - 42) > 0x11 )
                  goto LABEL_128;
                goto LABEL_90;
              }
              v43 = *(_BYTE **)(v39 - 64);
LABEL_56:
              if ( !v43 )
                goto LABEL_90;
            }
            if ( *v43 <= 0x1Cu )
              goto LABEL_90;
            goto LABEL_58;
          }
          if ( v41 )
          {
            if ( (_BYTE)v42 != 86 )
              goto LABEL_104;
            v43 = *(_BYTE **)(v39 - 64);
          }
          else
          {
            if ( (_BYTE)v42 != 86 )
            {
              if ( (unsigned int)(v42 - 42) > 0x11 )
                goto LABEL_128;
              v43 = *(_BYTE **)(v39 + 32LL * (unsigned int)(1 - v40) - 64);
              if ( !v43 )
                goto LABEL_90;
              goto LABEL_97;
            }
            v43 = *(_BYTE **)(v39 - 32);
          }
          if ( !v43 )
            goto LABEL_90;
LABEL_97:
          if ( *v43 <= 0x1Cu )
            goto LABEL_90;
          v78 = v34;
LABEL_58:
          v94 = 0;
          v95 = 0;
          v96 = 0;
          v97 = 0;
          v98 = 0;
          v99 = 0;
          v100 = 0;
          v101 = 0;
          v102 = 0;
          v103 = 0;
          sub_2785050((__int64 *)&v94, 0);
          sub_2F9CB00(v38, (__int64)v43, (unsigned __int64 *)&v94, v39, 0);
          v44 = v100;
          if ( v100 != v96 )
          {
            v69 = v37;
            v45 = 0;
            v46 = 0;
            do
            {
              v47 = *(__int64 ***)(v38 + 24);
              if ( v44 == v101 )
                v44 = *(v103 - 1) + 512;
              v48 = *(_QWORD *)(v44 - 8);
              v49 = 32LL * (*(_DWORD *)(v48 + 4) & 0x7FFFFFF);
              if ( (*(_BYTE *)(v48 + 7) & 0x40) != 0 )
              {
                v50 = *(_QWORD *)(v48 - 8);
                v51 = v50 + v49;
              }
              else
              {
                v50 = v48 - v49;
                v51 = *(_QWORD *)(v44 - 8);
              }
              v52 = v51 - v50;
              v83 = (unsigned __int8 **)v85;
              v84 = 0x400000000LL;
              v53 = v52 >> 5;
              v54 = v52 >> 5;
              if ( (unsigned __int64)v52 > 0x80 )
              {
                v74 = v52;
                v75 = v52 >> 5;
                v76 = v50;
                v77 = v48;
                v80 = v52 >> 5;
                sub_C8D5F0((__int64)&v83, v85, v53, 8u, v52, v48);
                v57 = v83;
                v56 = v84;
                LODWORD(v53) = v80;
                v48 = v77;
                v50 = v76;
                v54 = v75;
                v55 = &v83[(unsigned int)v84];
                v52 = v74;
              }
              else
              {
                v55 = (unsigned __int8 **)v85;
                v56 = 0;
                v57 = (unsigned __int8 **)v85;
              }
              if ( v52 > 0 )
              {
                v58 = 0;
                do
                {
                  v55[v58 / 8] = *(unsigned __int8 **)(v50 + 4 * v58);
                  v58 += 8LL;
                  --v54;
                }
                while ( v54 );
                v57 = v83;
                v56 = v84;
              }
              LODWORD(v84) = v53 + v56;
              v60 = sub_DFCEF0(v47, (unsigned __int8 *)v48, v57, (unsigned int)(v53 + v56), 1);
              if ( v83 != (unsigned __int8 **)v85 )
              {
                v79 = v59;
                _libc_free((unsigned __int64)v83);
                v59 = v79;
              }
              if ( v59 == 1 )
                v45 = 1;
              v61 = __OFADD__(v60, v46);
              v46 += v60;
              if ( v61 )
              {
                v46 = 0x8000000000000000LL;
                if ( v60 > 0 )
                  v46 = 0x7FFFFFFFFFFFFFFFLL;
              }
              if ( v100 == v101 )
              {
                j_j___libc_free_0(v100);
                v64 = *--v103 + 512;
                v101 = *v103;
                v44 = v101 + 504;
                v102 = v64;
                v100 = v101 + 504;
              }
              else
              {
                v44 = v100 - 8;
                v100 -= 8LL;
              }
            }
            while ( v96 != v44 );
            v37 = v69;
            v62 = v46 * v78;
            if ( is_mul_ok(v46, v78) )
            {
LABEL_79:
              v61 = __OFADD__(v68, v62);
              v63 = v68 + v62;
              if ( v61 )
              {
                v63 = 0x8000000000000000LL;
                if ( v68 )
                  v63 = 0x7FFFFFFFFFFFFFFFLL;
              }
            }
            else
            {
              if ( v46 <= 0 )
              {
                if ( v46 < 0 && v78 < 0 )
                {
                  v62 = 0x7FFFFFFFFFFFFFFFLL;
                  goto LABEL_79;
                }
              }
              else
              {
                v62 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v78 > 0 )
                  goto LABEL_79;
              }
              v63 = (v67 >> 1) + 0x8000000000000000LL;
            }
            if ( v45 )
              goto LABEL_81;
            goto LABEL_88;
          }
          v63 = v67 >> 1;
LABEL_88:
          if ( v63 / (__int64)v67 >= (unsigned int)(4 * qword_5025AE8) )
          {
LABEL_81:
            v2 = v38;
            sub_2784FD0((unsigned __int64 *)&v94);
            v18 = "Converted to branch because of expensive cold operand.";
            v19 = 54;
            goto LABEL_82;
          }
          sub_2784FD0((unsigned __int64 *)&v94);
LABEL_90:
          v37 += 2;
          if ( v73 == (unsigned __int8 **)v37 )
          {
            v2 = v38;
            break;
          }
          v34 = v81;
          v78 = v82;
        }
      }
    }
  }
  else if ( *(_QWORD *)(*(_QWORD *)(v2 + 48) + 8LL) )
  {
    sub_B176B0((__int64)&v94, (__int64)"select-optimize", (__int64)"SelectOpti", 10, **(_QWORD **)(a2 + 8));
    sub_B18290(
      (__int64)&v94,
      "Profile data available but missing branch-weights metadata for select instruction. ",
      0x53u);
    sub_1049740(*(__int64 **)(v2 + 56), (__int64)&v94);
    v21 = v104;
    v94 = &unk_49D9D40;
    v22 = &v104[10 * v105];
    if ( v104 != v22 )
    {
      do
      {
        v22 -= 10;
        v23 = v22[4];
        if ( (unsigned __int64 *)v23 != v22 + 6 )
          j_j___libc_free_0(v23);
        if ( (unsigned __int64 *)*v22 != v22 + 2 )
          j_j___libc_free_0(*v22);
      }
      while ( v21 != v22 );
      v22 = v104;
    }
    if ( v22 != (unsigned __int64 *)&v106 )
      _libc_free((unsigned __int64)v22);
  }
  v24 = *(_QWORD *)(*v71 + 40);
  v25 = *(_QWORD *)(v2 + 32);
  v26 = *(_DWORD *)(v25 + 24);
  v27 = *(_QWORD *)(v25 + 8);
  if ( !v26 )
    goto LABEL_45;
  v28 = v26 - 1;
  v29 = v28 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
  v30 = (__int64 *)(v27 + 16LL * v29);
  v31 = *v30;
  if ( v24 != *v30 )
  {
    v65 = 1;
    while ( v31 != -4096 )
    {
      v66 = v65 + 1;
      v29 = v28 & (v65 + v29);
      v30 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v30;
      if ( v24 == *v30 )
        goto LABEL_42;
      v65 = v66;
    }
    goto LABEL_45;
  }
LABEL_42:
  v32 = v30[1];
  if ( !v32 || *(_QWORD *)(v32 + 16) == *(_QWORD *)(v32 + 8) || v24 != sub_D47930(v32) || *(_DWORD *)(a2 + 16) <= 2u )
  {
LABEL_45:
    sub_B18290((__int64)v90, "Not profitable to convert to branch (base heuristic).", 0x35u);
    sub_1049740(*(__int64 **)(v2 + 56), (__int64)v90);
    v72 = 0;
    goto LABEL_5;
  }
  sub_B18290((__int64)v86, "Converted to branch because select group in the latch block is big.", 0x43u);
  sub_1049740(*(__int64 **)(v2 + 56), (__int64)v86);
  v72 = 1;
LABEL_5:
  v10 = v91;
  v90[0] = &unk_49D9D40;
  v11 = &v91[10 * v92];
  if ( v91 != v11 )
  {
    do
    {
      v11 -= 10;
      v12 = v11[4];
      if ( (unsigned __int64 *)v12 != v11 + 6 )
        j_j___libc_free_0(v12);
      if ( (unsigned __int64 *)*v11 != v11 + 2 )
        j_j___libc_free_0(*v11);
    }
    while ( v10 != v11 );
    v11 = v91;
  }
  if ( v11 != (unsigned __int64 *)&v93 )
    _libc_free((unsigned __int64)v11);
  v13 = v87;
  v86[0] = &unk_49D9D40;
  v14 = &v87[10 * v88];
  if ( v87 != v14 )
  {
    do
    {
      v14 -= 10;
      v15 = v14[4];
      if ( (unsigned __int64 *)v15 != v14 + 6 )
        j_j___libc_free_0(v15);
      if ( (unsigned __int64 *)*v14 != v14 + 2 )
        j_j___libc_free_0(*v14);
    }
    while ( v13 != v14 );
    v14 = v87;
  }
  if ( v14 != (unsigned __int64 *)&v89 )
    _libc_free((unsigned __int64)v14);
  return v72;
}
