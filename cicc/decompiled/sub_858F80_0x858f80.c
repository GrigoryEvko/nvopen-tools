// Function: sub_858F80
// Address: 0x858f80
//
_DWORD *__fastcall sub_858F80(unsigned __int64 a1, unsigned int *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  const char *v6; // rdx
  unsigned __int64 v7; // rsi
  unsigned int v8; // ebx
  int v9; // eax
  int v10; // r15d
  unsigned int *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  char *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  char v20; // al
  _BOOL4 v21; // eax
  _QWORD *v22; // r15
  __int64 v23; // r14
  __int64 i; // rdi
  int v25; // eax
  __int64 v26; // r14
  __int64 v28; // rdi
  __int64 v29; // r14
  size_t v30; // rax
  size_t v31; // r13
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rax
  __int64 **v34; // rax
  __int64 *v35; // rdx
  __int64 v36; // r13
  size_t v37; // rsi
  const void *v38; // rdi
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // r15
  unsigned __int64 v41; // r12
  unsigned __int64 v42; // rsi
  __int64 **v43; // rax
  __int64 *v44; // rcx
  __int64 *v45; // r13
  __int64 v46; // r9
  __int64 v47; // rsi
  __int64 **v48; // rax
  __int64 *v49; // rdx
  unsigned __int64 v50; // rdx
  __int64 *v51; // rdi
  __int64 v52; // rdi
  __int64 v53; // rsi
  __int64 v54; // r15
  __int64 v55; // r12
  _QWORD *v56; // rbx
  __int64 v57; // rdx
  _QWORD *v58; // r15
  bool v59; // cf
  bool v60; // zf
  const char *v61; // rdi
  unsigned int v62; // esi
  bool v63; // cf
  bool v64; // zf
  char v65; // dl
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // rdx
  unsigned __int64 v69; // rdx
  __int64 v70; // rdi
  unsigned __int64 v71; // r13
  __int64 **v72; // rax
  _QWORD *v73; // rdi
  __int64 *v74; // rax
  _BYTE *v75; // rsi
  size_t v76; // rdx
  __int64 v77; // rsi
  int *v78; // rdi
  char v79; // al
  unsigned __int64 v80; // rdx
  __int64 *v81; // r15
  __int64 *v82; // r10
  unsigned __int64 v83; // rcx
  __int64 v84; // r13
  __int64 *v85; // rdx
  __int64 v86; // r13
  __int64 v87; // rdx
  __int64 v88; // rax
  size_t v89; // r13
  _QWORD *v90; // r8
  unsigned __int64 v91; // r9
  _QWORD *v92; // rsi
  unsigned __int64 v93; // rdx
  __int64 *v94; // rax
  unsigned __int64 v95; // r13
  void *v96; // rdi
  __int64 *v97; // r13
  __int64 *v98; // [rsp+8h] [rbp-98h]
  unsigned __int64 v99; // [rsp+8h] [rbp-98h]
  __int64 v100; // [rsp+8h] [rbp-98h]
  __int64 *v101; // [rsp+10h] [rbp-90h]
  __int64 v102; // [rsp+10h] [rbp-90h]
  unsigned __int64 v103; // [rsp+10h] [rbp-90h]
  __int64 *v104; // [rsp+10h] [rbp-90h]
  unsigned int v105; // [rsp+18h] [rbp-88h]
  unsigned int v106; // [rsp+18h] [rbp-88h]
  unsigned __int64 v107; // [rsp+18h] [rbp-88h]
  unsigned __int64 v108; // [rsp+18h] [rbp-88h]
  unsigned int v109; // [rsp+20h] [rbp-80h]
  __int64 v110; // [rsp+20h] [rbp-80h]
  _BOOL4 v111; // [rsp+2Ch] [rbp-74h]
  char *s2; // [rsp+30h] [rbp-70h]
  unsigned int v113; // [rsp+38h] [rbp-68h]
  size_t v114; // [rsp+48h] [rbp-58h] BYREF
  void *v115; // [rsp+50h] [rbp-50h] BYREF
  size_t v116; // [rsp+58h] [rbp-48h]
  _QWORD v117[8]; // [rsp+60h] [rbp-40h] BYREF

  v113 = a1;
  if ( sub_7B0640() )
    unk_4D03CA8 = 1;
  dword_4D03CF8 = 1;
  if ( !(_DWORD)a1 )
  {
    dword_4D03D1C = 1;
    if ( (unsigned __int16)sub_7B8B50(a1, a2, v2, v3, v4, v5) != 13 )
    {
      sub_6851D0(0x21u);
      dword_4D03CE0 = 1;
      goto LABEL_38;
    }
    v6 = qword_4F06410;
    v7 = qword_4F06408;
    if ( qword_4F06408 < (unsigned __int64)qword_4F06410 )
    {
      v8 = 0;
      goto LABEL_11;
    }
LABEL_5:
    v8 = 0;
    v9 = *v6 - 48;
    while ( 1 )
    {
      ++v6;
      v8 += v9;
      if ( (unsigned __int64)v6 > v7 )
        break;
      if ( v8 > 0x19999999 )
        goto LABEL_11;
      v8 *= 10;
      v9 = *v6 - 48;
      v3 = (unsigned int)~v9;
      if ( (unsigned int)v3 < v8 )
      {
        if ( !v8 )
          v8 = a1;
        goto LABEL_11;
      }
    }
    v10 = 0;
    if ( v8 )
      goto LABEL_12;
    if ( (_DWORD)a1 )
      goto LABEL_84;
LABEL_11:
    v7 = (unsigned __int64)dword_4F07508;
    a1 = 34;
    v10 = 1;
    sub_6851C0(0x22u, dword_4F07508);
    goto LABEL_12;
  }
  v6 = qword_4F06410;
  v7 = qword_4F06408;
  if ( qword_4F06408 >= (unsigned __int64)qword_4F06410 )
    goto LABEL_5;
LABEL_84:
  v10 = 0;
  v8 = 1;
LABEL_12:
  if ( (unsigned __int16)sub_7B8B50(a1, (unsigned int *)v7, (__int64)v6, v3, v4, v5) == 10 )
  {
    v15 = (char *)qword_4F064B0[1];
    s2 = v15;
    v111 = (*(_BYTE *)(qword_4F064B0[7] + 72LL) & 0x40) != 0;
    if ( v113 )
      goto LABEL_16;
LABEL_43:
    if ( v10 )
      goto LABEL_38;
    v22 = qword_4F064B0;
    v28 = qword_4F064B0[7];
    if ( v28 != qword_4F064B0[8] )
    {
      sub_729A00(v28, unk_4F06468);
      v22 = qword_4F064B0;
LABEL_106:
      v110 = v22[8];
      v29 = v110;
      goto LABEL_46;
    }
    v110 = qword_4F064B0[7];
    v29 = v110;
LABEL_46:
    v22[1] = s2;
    *((_DWORD *)v22 + 10) = v8 - 1;
    if ( !dword_4D0460C || (v113 & 1) != 0 || !s2 )
    {
LABEL_77:
      sub_729880(
        v110,
        unk_4F06468 + 1,
        v8,
        (__int64)s2,
        0,
        0,
        v22 + 7,
        (*(_BYTE *)(v29 + 72) & 4) != 0,
        (*(_BYTE *)(v29 + 72) & 8) != 0,
        (*(_BYTE *)(v29 + 72) & 0x10) != 0,
        (*(_BYTE *)(v29 + 72) & 0x20) != 0,
        (*(_BYTE *)(v29 + 72) & 2) != 0,
        v111 || (*(_BYTE *)(v29 + 72) & 0x40) != 0);
      goto LABEL_78;
    }
    v115 = v117;
    v30 = strlen(s2);
    v114 = v30;
    v31 = v30;
    if ( v30 > 0xF )
    {
      v115 = (void *)sub_22409D0(&v115, &v114, 0);
      v73 = v115;
      v117[0] = v114;
    }
    else
    {
      if ( v30 == 1 )
      {
        LOBYTE(v117[0]) = *s2;
        v32 = v117;
LABEL_52:
        v116 = v30;
        *((_BYTE *)v32 + v30) = 0;
        v33 = sub_22076E0(v115, v116, 3339675911LL);
        v34 = sub_858ED0(&qword_4F5FC20, v33 % qword_4F5FC28, (__int64)&v115, v33);
        v35 = (__int64 *)qword_4F5FC90;
        if ( v34 )
        {
          v36 = qword_4F5FC90;
          v35 = (__int64 *)qword_4F5FC90;
          if ( *v34 )
          {
            if ( qword_4F5FC90 == qword_4F5FC70 )
              goto LABEL_133;
            v106 = v8;
            while ( 1 )
            {
              v54 = qword_4F5FC98;
              v55 = qword_4F5FCA8;
              v56 = v115;
              if ( qword_4F5FC98 == v36 )
              {
                v57 = *(_QWORD *)(qword_4F5FCA8 - 8);
                v37 = *(_QWORD *)(v57 + 488);
                v38 = *(const void **)(v57 + 480);
                if ( v116 != v37 )
                  goto LABEL_111;
              }
              else
              {
                v37 = *(_QWORD *)(v36 - 24);
                v38 = *(const void **)(v36 - 32);
                if ( v37 != v116 )
                  goto LABEL_57;
              }
              if ( !v37 || !memcmp(v38, v115, v37) )
              {
                v58 = v56;
                v8 = v106;
                goto LABEL_74;
              }
              if ( v54 == v36 )
              {
                v57 = *(_QWORD *)(v55 - 8);
                v37 = *(_QWORD *)(v57 + 488);
                v38 = *(const void **)(v57 + 480);
LABEL_111:
                v36 = v57 + 512;
                goto LABEL_57;
              }
              v37 = *(_QWORD *)(v36 - 24);
              v38 = *(const void **)(v36 - 32);
LABEL_57:
              v39 = sub_22076E0(v38, v37, 3339675911LL);
              v40 = qword_4F5FC28;
              v41 = v39 % qword_4F5FC28;
              v42 = v39 % qword_4F5FC28;
              v43 = sub_858ED0(&qword_4F5FC20, v39 % qword_4F5FC28, v36 - 32, v39);
              v44 = (__int64 *)v43;
              if ( v43 )
              {
                v45 = *v43;
                v46 = 8 * v41;
                v47 = **v43;
                v48 = (__int64 **)(qword_4F5FC20 + 8 * v41);
                v49 = *v48;
                if ( v44 != *v48 )
                {
                  if ( v47 )
                  {
                    v50 = *(_QWORD *)(v47 + 40) % v40;
                    if ( v41 != v50 )
                    {
                      *(_QWORD *)(qword_4F5FC20 + 8 * v50) = v44;
                      v47 = *v45;
                    }
                  }
                  goto LABEL_62;
                }
                if ( v47 )
                {
                  v69 = *(_QWORD *)(v47 + 40) % v40;
                  if ( v41 == v69 )
                  {
LABEL_62:
                    *v44 = v47;
                    v51 = (__int64 *)v45[1];
                    if ( v51 != v45 + 3 )
                      j_j___libc_free_0(v51, v45[3] + 1);
                    v42 = 48;
                    j_j___libc_free_0(v45, 48);
                    --qword_4F5FC38;
                    goto LABEL_65;
                  }
                  *(_QWORD *)(qword_4F5FC20 + 8 * v69) = v44;
                  v48 = (__int64 **)(v46 + qword_4F5FC20);
                  v49 = *(__int64 **)(v46 + qword_4F5FC20);
                }
                if ( v49 == &qword_4F5FC30 )
                  qword_4F5FC30 = v47;
                *v48 = 0;
                v47 = *v45;
                goto LABEL_62;
              }
LABEL_65:
              v52 = qword_4F5FC90;
              if ( qword_4F5FC90 == qword_4F5FC98 )
              {
                v42 = 512;
                j_j___libc_free_0(qword_4F5FC90, 512);
                qword_4F5FCA8 -= 8;
                v67 = *(_QWORD *)qword_4F5FCA8;
                v68 = *(_QWORD *)qword_4F5FCA8 + 512LL;
                v52 = *(_QWORD *)(*(_QWORD *)qword_4F5FCA8 + 480LL);
                qword_4F5FC98 = v67;
                qword_4F5FCA0 = v68;
                qword_4F5FC90 = v67 + 480;
                if ( v52 != v67 + 496 )
                {
                  v42 = *(_QWORD *)(v67 + 496) + 1LL;
                  j_j___libc_free_0(v52, v42);
                }
              }
              else
              {
                qword_4F5FC90 -= 32;
                if ( *(_QWORD *)(v52 - 32) != v52 - 16 )
                {
                  v53 = *(_QWORD *)(v52 - 16);
                  v52 = *(_QWORD *)(v52 - 32);
                  v42 = v53 + 1;
                  j_j___libc_free_0(v52, v42);
                }
              }
              unk_4D045D0(v52, v42);
              v36 = qword_4F5FC90;
              if ( qword_4F5FC90 == qword_4F5FC70 )
              {
                v8 = v106;
LABEL_133:
                v58 = v115;
                goto LABEL_74;
              }
            }
          }
        }
        v70 = (__int64)v35;
        if ( v35 != (__int64 *)(qword_4F5FCA0 - 32) )
        {
          if ( v35 )
          {
            *v35 = (__int64)(v35 + 2);
            sub_856170(v35, v115, (__int64)v115 + v116);
            v70 = qword_4F5FC90;
          }
          qword_4F5FC90 = v70 + 32;
LABEL_128:
          v107 = sub_22076E0(v115, v116, 3339675911LL);
          v71 = v107 % qword_4F5FC28;
          v72 = sub_858ED0(&qword_4F5FC20, v107 % qword_4F5FC28, (__int64)&v115, v107);
          if ( v72 && *v72 )
            goto LABEL_130;
          v74 = (__int64 *)sub_22077B0(48);
          if ( v74 )
            *v74 = 0;
          v75 = v115;
          v76 = v116;
          v74[1] = (__int64)(v74 + 3);
          v101 = v74;
          sub_856170(v74 + 1, v75, (__int64)&v75[v76]);
          v77 = qword_4F5FC28;
          v78 = &dword_4F5FC40;
          v79 = sub_222DA10(&dword_4F5FC40, qword_4F5FC28, qword_4F5FC38, 1);
          v81 = (__int64 *)qword_4F5FC20;
          v82 = v101;
          v83 = v80;
          if ( !v79 )
          {
LABEL_141:
            v84 = v71;
            v82[5] = v107;
            v85 = (__int64 *)v81[v84];
            if ( v85 )
            {
              *v82 = *v85;
              *(_QWORD *)v81[v84] = v82;
            }
            else
            {
              v88 = qword_4F5FC30;
              qword_4F5FC30 = (__int64)v82;
              *v82 = v88;
              if ( v88 )
                v81[*(_QWORD *)(v88 + 40) % (unsigned __int64)qword_4F5FC28] = (__int64)v82;
              *(_QWORD *)(qword_4F5FC20 + v84 * 8) = &qword_4F5FC30;
            }
            ++qword_4F5FC38;
LABEL_130:
            if ( qword_4F064B0[7] == qword_4F064B0[8] )
              goto LABEL_133;
            unk_4D045D8("Processing Header File", s2);
            v58 = v115;
LABEL_74:
            if ( v58 != v117 )
              j_j___libc_free_0(v58, v117[0] + 1LL);
            v22 = qword_4F064B0;
            goto LABEL_77;
          }
          if ( v80 == 1 )
          {
            qword_4F5FC50 = 0;
            v81 = &qword_4F5FC50;
            goto LABEL_155;
          }
          v98 = v101;
          if ( v80 <= 0xFFFFFFFFFFFFFFFLL )
          {
            v89 = 8 * v80;
            v103 = v80;
            v81 = (__int64 *)sub_22077B0(8 * v80);
            memset(v81, 0, v89);
            v82 = v98;
            v83 = v103;
LABEL_155:
            v90 = (_QWORD *)qword_4F5FC30;
            qword_4F5FC30 = 0;
            if ( v90 )
            {
              v91 = 0;
              do
              {
                v92 = v90;
                v90 = (_QWORD *)*v90;
                v93 = v92[5] % v83;
                v94 = &v81[v93];
                if ( *v94 )
                {
                  *v92 = *(_QWORD *)*v94;
                  *(_QWORD *)*v94 = v92;
                }
                else
                {
                  *v92 = qword_4F5FC30;
                  qword_4F5FC30 = (__int64)v92;
                  *v94 = (__int64)&qword_4F5FC30;
                  if ( *v92 )
                    v81[v91] = (__int64)v92;
                  v91 = v93;
                }
              }
              while ( v90 );
            }
            if ( (__int64 *)qword_4F5FC20 != &qword_4F5FC50 )
            {
              v99 = v83;
              v104 = v82;
              j_j___libc_free_0(qword_4F5FC20, 8 * qword_4F5FC28);
              v83 = v99;
              v82 = v104;
            }
            qword_4F5FC28 = v83;
            qword_4F5FC20 = (__int64)v81;
            v71 = v107 % v83;
            goto LABEL_141;
          }
LABEL_167:
          sub_4261EA(v78, v77, v80, v83);
        }
        v78 = (int *)qword_4F5FCA8;
        v102 = qword_4F5FCA8 - (_QWORD)qword_4F5FC88;
        v83 = (qword_4F5FCA8 - (__int64)qword_4F5FC88) >> 3;
        if ( ((qword_4F5FC80 - qword_4F5FC70) >> 5) + 16 * (v83 - 1) + (((__int64)v35 - qword_4F5FC98) >> 5) == 0x3FFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
        if ( (unsigned __int64)(qword_4F5FC68 - ((qword_4F5FCA8 - qword_4F5FC60) >> 3)) > 1 )
        {
LABEL_146:
          v86 = qword_4F5FCA8;
          *(_QWORD *)(v86 + 8) = sub_22077B0(512);
          if ( qword_4F5FC90 )
            sub_2241BD0(qword_4F5FC90, &v115);
          qword_4F5FCA8 += 8;
          v87 = *(_QWORD *)qword_4F5FCA8 + 512LL;
          qword_4F5FC98 = *(_QWORD *)qword_4F5FCA8;
          qword_4F5FCA0 = v87;
          qword_4F5FC90 = qword_4F5FC98;
          goto LABEL_128;
        }
        v95 = v83 + 2;
        if ( qword_4F5FC68 <= 2 * (v83 + 2) )
        {
          v80 = 1;
          if ( qword_4F5FC68 )
            v80 = qword_4F5FC68;
          v108 = qword_4F5FC68 + v80 + 2;
          v77 = v108;
          if ( v108 > 0xFFFFFFFFFFFFFFFLL )
            goto LABEL_167;
          v100 = sub_22077B0(8 * v108);
          v97 = (__int64 *)(v100 + 8 * ((v108 - v95) >> 1));
          if ( (void *)(qword_4F5FCA8 + 8) != qword_4F5FC88 )
            memmove(v97, qword_4F5FC88, qword_4F5FCA8 + 8 - (_QWORD)qword_4F5FC88);
          j_j___libc_free_0(qword_4F5FC60, 8 * qword_4F5FC68);
          qword_4F5FC60 = v100;
          qword_4F5FC68 = v108;
          goto LABEL_173;
        }
        v96 = (void *)(qword_4F5FCA8 + 8);
        v97 = (__int64 *)(qword_4F5FC60 + 8 * ((qword_4F5FC68 - v95) >> 1));
        if ( qword_4F5FC88 <= v97 )
        {
          if ( qword_4F5FC88 == v96 )
            goto LABEL_173;
        }
        else if ( qword_4F5FC88 == v96 )
        {
LABEL_173:
          qword_4F5FC88 = v97;
          qword_4F5FC78 = *v97;
          qword_4F5FC80 = qword_4F5FC78 + 512;
          qword_4F5FCA8 = (__int64)v97 + v102;
          qword_4F5FC98 = *(__int64 *)((char *)v97 + v102);
          qword_4F5FCA0 = qword_4F5FC98 + 512;
          goto LABEL_146;
        }
        memmove(v97, qword_4F5FC88, qword_4F5FCA8 + 8 - (_QWORD)qword_4F5FC88);
        goto LABEL_173;
      }
      if ( !v30 )
      {
        v32 = v117;
        goto LABEL_52;
      }
      v73 = v117;
    }
    memcpy(v73, s2, v31);
    v30 = v114;
    v32 = v115;
    goto LABEL_52;
  }
  if ( word_4F06418[0] != 7 || *qword_4F06410 == 76 )
  {
    sub_6851D0(0xDu);
    dword_4D03CE0 = 1;
    goto LABEL_38;
  }
  v15 = (char *)(dword_4D04954 == 0);
  s2 = (char *)sub_857470((int)v15);
  sub_7B8B50((unsigned __int64)v15, (unsigned int *)v7, v16, v17, v18, v19);
  v111 = 0;
  if ( !v113 )
    goto LABEL_43;
LABEL_16:
  v109 = 0;
  v105 = 0;
  while ( word_4F06418[0] == 13 )
  {
    while ( qword_4F06400 != 1 )
    {
LABEL_18:
      sub_7B8B50((unsigned __int64)v15, (unsigned int *)v7, (__int64)v11, v12, v13, v14);
      if ( word_4F06418[0] != 13 )
        goto LABEL_25;
    }
    v20 = *qword_4F06410;
    if ( *qword_4F06410 == 49 )
    {
      v14 = dword_4D0460C;
      if ( dword_4D0460C && (v11 = (unsigned int *)s2, v7 = (unsigned __int64)s2, v59 = 0, v60 = s2 == 0, s2) )
      {
        v61 = "<built-in>";
        v12 = 11;
        do
        {
          if ( !v12 )
            break;
          v59 = *(_BYTE *)v7 < *v61;
          v60 = *(_BYTE *)v7++ == *v61++;
          --v12;
        }
        while ( v60 );
        v15 = (char *)v113;
        v109 = v113;
        if ( (!v59 && !v60) != v59 )
        {
          v7 = (unsigned __int64)s2;
          v15 = "Processing Header File";
          unk_4D045D8("Processing Header File", s2);
        }
      }
      else
      {
        v109 = v113;
      }
      goto LABEL_18;
    }
    if ( v20 == 50 )
    {
      v15 = (char *)v113;
      v11 = &dword_4D0460C;
      v13 = dword_4D0460C;
      v62 = v109;
      if ( qword_4F064B0[7] == qword_4F064B0[8] )
        v62 = v113;
      v109 = v62;
      v7 = v105;
      if ( qword_4F064B0[7] != qword_4F064B0[8] )
        v7 = v113;
      v105 = v7;
      if ( dword_4D0460C )
      {
        v63 = 0;
        v64 = s2 == 0;
        if ( s2 )
        {
          v7 = qword_4F064B0[1];
          v12 = 11;
          v15 = "<built-in>";
          do
          {
            if ( !v12 )
              break;
            v63 = *(_BYTE *)v7 < (unsigned __int8)*v15;
            v64 = *(_BYTE *)v7++ == (unsigned __int8)*v15++;
            --v12;
          }
          while ( v64 );
          if ( (!v63 && !v64) != v63 )
            unk_4D045D0(v15, v7);
        }
      }
      goto LABEL_18;
    }
    v60 = v20 == 51;
    v21 = v111;
    if ( v60 )
      v21 = v113;
    v111 = v21;
    sub_7B8B50((unsigned __int64)v15, (unsigned int *)v7, (__int64)v11, v12, v13, v14);
  }
LABEL_25:
  if ( !v10 )
  {
    v22 = qword_4F064B0;
    v23 = qword_4F064B0[7];
    if ( v23 != qword_4F064B0[8] )
    {
      if ( !strcmp((const char *)qword_4F064B0[1], s2) )
      {
        sub_729A40(0, unk_4F06468 + 1, v8);
        v22 = qword_4F064B0;
        goto LABEL_108;
      }
      if ( (v109 & 1) == 0 )
      {
        for ( i = v23; ; i = v22[7] )
        {
          sub_729A00(i, unk_4F06468);
          v22 = qword_4F064B0;
          v25 = strcmp((const char *)qword_4F064B0[1], s2);
          if ( (v22[11] & 0x40) == 0 )
            break;
          if ( !v25 )
            goto LABEL_35;
          sub_7B0E20();
          v22 = qword_4F064B0;
          if ( strcmp((const char *)qword_4F064B0[1], s2) )
            goto LABEL_106;
        }
        if ( !v25 )
        {
LABEL_35:
          sub_729A40(0, unk_4F06468 + 1, v8);
          v22 = qword_4F064B0;
          if ( !v105 )
          {
            v26 = qword_4F064B0[8];
            qword_4F064B0[1] = s2;
            *((_DWORD *)v22 + 10) = v8 - 1;
            sub_729880(
              v26,
              unk_4F06468 + 1,
              v8,
              (__int64)s2,
              0,
              0,
              v22 + 7,
              (*(_BYTE *)(v26 + 72) & 4) != 0,
              (*(_BYTE *)(v26 + 72) & 8) != 0,
              (*(_BYTE *)(v26 + 72) & 0x10) != 0,
              (*(_BYTE *)(v26 + 72) & 0x20) != 0,
              (*(_BYTE *)(v26 + 72) & 2) != 0,
              v111 || (*(_BYTE *)(v26 + 72) & 0x40) != 0);
LABEL_78:
            if ( dword_4D0493C )
              sub_7AF280(32, 1);
            if ( qword_4D04908 )
              sub_7AF3F0(32);
            goto LABEL_38;
          }
LABEL_108:
          v65 = *((_BYTE *)v22 + 88);
          *((_DWORD *)v22 + 10) = v8 - 1;
          v22[1] = s2;
          v66 = v22[7];
          *((_BYTE *)v22 + 88) = (2 * v111) | v65 & 0xFD;
          *(_BYTE *)(v66 + 72) = (v111 << 6) | *(_BYTE *)(v66 + 72) & 0xBF;
          goto LABEL_78;
        }
        goto LABEL_106;
      }
    }
    if ( v105 )
      goto LABEL_108;
    v29 = v22[8];
    if ( v109 )
    {
      sub_7B0C00();
      v22 = qword_4F064B0;
      v110 = qword_4F064B0[7];
    }
    else
    {
      v110 = v22[8];
    }
    goto LABEL_46;
  }
LABEL_38:
  dword_4D03CF8 = 0;
  return &dword_4D03CF8;
}
