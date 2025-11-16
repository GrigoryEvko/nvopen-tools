// Function: sub_6D3EC0
// Address: 0x6d3ec0
//
__int64 __fastcall sub_6D3EC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  _BYTE *v5; // r14
  __int64 v6; // r12
  _QWORD *v7; // rbx
  __int64 *v8; // r12
  __int64 v9; // r8
  int v10; // r13d
  __int64 v11; // r13
  __int64 *v12; // r14
  __int64 v13; // rax
  __int64 v14; // r12
  char v15; // al
  int v16; // eax
  __int64 v17; // rdi
  __int64 i; // rdx
  _DWORD *v19; // rsi
  _DWORD *v20; // rax
  int v21; // eax
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 j; // rax
  __int64 v29; // rdi
  int v30; // eax
  int v31; // r8d
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned __int16 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rbx
  __int64 v43; // r12
  __int64 v44; // rdx
  __int64 v45; // rcx
  int v46; // eax
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // r14
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdi
  int v54; // r15d
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rbx
  __int64 *v60; // rax
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rax
  __int64 v64; // rbx
  __int64 *v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  unsigned int v70; // r13d
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rcx
  __int64 v75; // r13
  __int64 v76; // rbx
  int v77; // eax
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned int v80; // [rsp+8h] [rbp-648h]
  __int64 v81; // [rsp+10h] [rbp-640h]
  __int64 v82; // [rsp+10h] [rbp-640h]
  _DWORD *v83; // [rsp+18h] [rbp-638h]
  unsigned int v84; // [rsp+20h] [rbp-630h]
  int v85; // [rsp+20h] [rbp-630h]
  int v86; // [rsp+20h] [rbp-630h]
  int v87; // [rsp+20h] [rbp-630h]
  __int64 v88; // [rsp+20h] [rbp-630h]
  __int64 v89; // [rsp+20h] [rbp-630h]
  _QWORD *v90; // [rsp+20h] [rbp-630h]
  __int64 v91; // [rsp+20h] [rbp-630h]
  __int64 v92; // [rsp+28h] [rbp-628h]
  __int64 v93; // [rsp+28h] [rbp-628h]
  int v94; // [rsp+30h] [rbp-620h]
  __int64 v95; // [rsp+38h] [rbp-618h]
  int v96; // [rsp+44h] [rbp-60Ch]
  unsigned int v97; // [rsp+44h] [rbp-60Ch]
  __int16 v98; // [rsp+62h] [rbp-5EEh]
  unsigned __int16 v99; // [rsp+62h] [rbp-5EEh]
  int v100; // [rsp+64h] [rbp-5ECh]
  int v101; // [rsp+64h] [rbp-5ECh]
  __int64 v102; // [rsp+68h] [rbp-5E8h]
  __int64 v103; // [rsp+68h] [rbp-5E8h]
  __int16 v104; // [rsp+68h] [rbp-5E8h]
  int v105; // [rsp+68h] [rbp-5E8h]
  char v106; // [rsp+7Ch] [rbp-5D4h] BYREF
  int v107; // [rsp+80h] [rbp-5D0h] BYREF
  unsigned int v108; // [rsp+84h] [rbp-5CCh] BYREF
  __int64 v109; // [rsp+88h] [rbp-5C8h] BYREF
  __int64 v110; // [rsp+90h] [rbp-5C0h] BYREF
  __int64 v111; // [rsp+98h] [rbp-5B8h] BYREF
  __int64 v112[8]; // [rsp+A0h] [rbp-5B0h] BYREF
  _DWORD v113[71]; // [rsp+E4h] [rbp-56Ch] BYREF
  _BYTE v114[352]; // [rsp+200h] [rbp-450h] BYREF
  _QWORD v115[44]; // [rsp+360h] [rbp-2F0h] BYREF
  _BYTE v116[68]; // [rsp+4C0h] [rbp-190h] BYREF
  __int64 v117; // [rsp+504h] [rbp-14Ch]
  __int64 v118; // [rsp+50Ch] [rbp-144h]
  __int64 v119; // [rsp+540h] [rbp-110h]

  v4 = a2;
  v5 = (_BYTE *)a1;
  if ( !a1 )
  {
    v35 = word_4F06418[0];
    v100 = word_4F06418[0] != 257;
    v111 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, a2, a3, a4);
    sub_7BE280(27, 125, 0, 0);
    v36 = qword_4F061C8;
    v37 = qword_4D03C50;
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(v37 + 40);
    ++*(_BYTE *)(v36 + 75);
    sub_69ED20((__int64)v112, 0, 0, 1);
    sub_7BE280(67, 253, 0, 0);
    sub_69ED20((__int64)v114, 0, 0, 1);
    --*(_BYTE *)(qword_4F061C8 + 75LL);
    if ( word_4F06418[0] != 67 )
    {
      v102 = 0;
      v95 = 0;
      if ( v35 == 257 )
      {
        v7 = v114;
        v8 = 0;
        v92 = 0;
LABEL_69:
        v96 = qword_4F063F0;
        v98 = WORD2(qword_4F063F0);
        sub_7BE280(28, 18, 0, 0);
        --*(_BYTE *)(qword_4F061C8 + 36LL);
        --*(_QWORD *)(qword_4D03C50 + 40LL);
        goto LABEL_24;
      }
      v7 = 0;
LABEL_123:
      v8 = (__int64 *)v114;
      v92 = 0;
      goto LABEL_69;
    }
    if ( v35 == 257 )
    {
      sub_7B8B50(v114, 0, v38, v39);
      v7 = v115;
      sub_69ED20((__int64)v115, 0, 0, 1);
      v102 = 0;
      v95 = 0;
      goto LABEL_123;
    }
    v42 = 0;
    v43 = 0;
    v110 = sub_724DC0(v114, 0, v38, v39, v40, v41);
    sub_7B8B50(v114, 0, v44, v45);
    v93 = 0;
    v95 = 0;
    v94 = 0;
    ++*(_BYTE *)(qword_4F061C8 + 75LL);
    while ( 1 )
    {
      v46 = sub_869470(v115);
      v97 = dword_4F063F8;
      v99 = word_4F063FC[0];
      if ( v46 )
        break;
LABEL_102:
      if ( !(unsigned int)sub_7BE800(67) )
      {
        v102 = v42;
        v92 = v43;
        v5 = 0;
        v4 = a2;
        --*(_BYTE *)(qword_4F061C8 + 75LL);
        sub_724E30(&v110);
        v7 = v115;
        v8 = (__int64 *)v114;
        goto LABEL_69;
      }
    }
    while ( 1 )
    {
      sub_6BA680(v110);
      v50 = v110;
      v51 = (unsigned int)dword_4F077C8;
      v52 = *(unsigned int *)(v110 + 64);
      if ( (_DWORD)v52 == dword_4F077C8 )
      {
        v52 = *(unsigned __int16 *)(v110 + 68);
        v51 = unk_4F077CC;
      }
      if ( v52 == v51 )
      {
        v50 = sub_740630(v110);
        *(_DWORD *)(v50 + 64) = v97;
        *(_WORD *)(v50 + 68) = v99;
      }
      v53 = *(_QWORD *)(v50 + 128);
      v54 = *(_DWORD *)(v50 + 112);
      v55 = *(unsigned __int8 *)(v53 + 140);
      v104 = *(_WORD *)(v50 + 116);
      if ( (_BYTE)v55 == 12 )
      {
        v56 = *(_QWORD *)(v50 + 128);
        do
        {
          v56 = *(_QWORD *)(v56 + 160);
          v55 = *(unsigned __int8 *)(v56 + 140);
        }
        while ( (_BYTE)v55 == 12 );
      }
      if ( (_BYTE)v55 )
      {
        if ( (dword_4F04C44 != -1
           || (v57 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v57 + 6) & 6) != 0)
           || *(_BYTE *)(v57 + 4) == 12)
          && (unsigned int)sub_8DBE70(v53) )
        {
          if ( !v42 )
          {
            sub_6E6A50(v50, v116);
            LODWORD(v118) = v54;
            LODWORD(v117) = v97;
            WORD2(v117) = v99;
            *(_QWORD *)dword_4F07508 = v117;
            WORD2(v118) = v104;
            *(_QWORD *)&dword_4F061D8 = v118;
            sub_6E3280(v116, 0);
            goto LABEL_119;
          }
          v94 = 1;
        }
        else if ( !v42 )
        {
          goto LABEL_114;
        }
      }
      else
      {
        sub_6E6000(v53, 0, v55, v47, v48, v49);
        v42 = sub_72C930(v53);
        if ( !v42 )
        {
LABEL_114:
          sub_6E6A50(v50, v116);
          LODWORD(v118) = v54;
          LODWORD(v117) = v97;
          WORD2(v117) = v99;
          *(_QWORD *)dword_4F07508 = v117;
          WORD2(v118) = v104;
          *(_QWORD *)&dword_4F061D8 = v118;
          sub_6E3280(v116, 0);
          if ( !v94 )
          {
            sub_6F69D0(v116, 0);
            goto LABEL_116;
          }
LABEL_119:
          sub_6F40C0(v116);
          v94 = 1;
LABEL_116:
          v61 = sub_6F6F40(v116, 0);
          if ( v95 )
          {
            v62 = v93;
            v42 = 0;
            v93 = v61;
            *(_QWORD *)(v62 + 16) = v61;
          }
          else
          {
            v93 = v61;
            v42 = 0;
            v95 = v61;
          }
        }
      }
      ++v43;
      v58 = sub_867630(v115[0], 0);
      if ( v58 )
        v119 = v58;
      if ( !(unsigned int)sub_866C00(v115[0]) )
        goto LABEL_102;
    }
  }
  v6 = *(_QWORD *)a1;
  v7 = v115;
  v100 = *(_BYTE *)(*(_QWORD *)a1 + 56LL) != 53;
  v96 = *(_DWORD *)(*(_QWORD *)a1 + 44LL);
  v98 = *(_WORD *)(*(_QWORD *)a1 + 48LL);
  sub_6F8AB0(
    a1,
    (unsigned int)v112,
    (unsigned int)v114,
    (unsigned int)v115,
    (unsigned int)&v111,
    (unsigned int)&v106,
    0);
  v8 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v6 + 64) + 16LL) + 16LL);
  if ( v8 )
  {
    v8 = (__int64 *)v114;
    v92 = 0;
    v102 = 0;
    LODWORD(v9) = sub_688510(a1, v112, &v107);
    v95 = 0;
    if ( (_DWORD)v9 )
      goto LABEL_26;
    goto LABEL_4;
  }
  v7 = v114;
  v102 = 0;
  v95 = 0;
  v92 = 0;
LABEL_24:
  a1 = (__int64)v5;
  LODWORD(v9) = sub_688510((__int64)v5, v112, &v107);
  if ( (_DWORD)v9 )
  {
    if ( !v8 )
      goto LABEL_7;
LABEL_26:
    v84 = v9;
    v16 = sub_688510((__int64)v5, v8, &v108);
    v17 = v112[0];
    LODWORD(v9) = v84;
    v10 = v16;
    for ( i = v112[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v19 = (_DWORD *)*v8;
    v20 = (_DWORD *)*v8;
    if ( *(_BYTE *)(*v8 + 140) == 12 )
    {
      do
        v20 = (_DWORD *)*((_QWORD *)v20 + 20);
      while ( *((_BYTE *)v20 + 140) == 12 );
    }
    if ( v84 )
    {
      if ( v10 )
      {
        if ( (_DWORD *)v112[0] != v19 )
        {
          v81 = *(_QWORD *)(i + 160);
          v83 = (_DWORD *)*((_QWORD *)v20 + 20);
          v21 = sub_8D97D0(v112[0], v19, 32, v81, v84);
          v9 = v84;
          if ( !v21 )
          {
            if ( v100 )
            {
              if ( !v7 )
              {
                v17 = (__int64)v83;
                v77 = sub_8D2780(v83);
                v9 = v84;
                if ( v77 )
                {
                  v80 = v84;
                  v78 = sub_8D4620(v112[0]);
                  v17 = *v8;
                  v91 = v78;
                  v79 = sub_8D4620(*v8);
                  v9 = v80;
                  if ( v91 == v79 )
                    goto LABEL_42;
                }
              }
              if ( (_DWORD *)v81 == v83
                || (v19 = v83, v17 = v81, v85 = v9, v22 = sub_8D97D0(v81, v83, 32, v83, v9), LODWORD(v9) = v85, v22) )
              {
LABEL_42:
                if ( v102 == sub_72C930(v17) )
                  goto LABEL_9;
                if ( v107 )
                {
                  if ( !v102 )
                    goto LABEL_49;
                  goto LABEL_46;
                }
                if ( v108 )
                {
                  if ( !v102 )
                    goto LABEL_11;
                  goto LABEL_46;
                }
                v109 = sub_724DC0(v17, v19, v24, v25, v26, v108);
                v110 = sub_724DC0(v17, v19, v66, v67, v68, v69);
                v70 = unk_4F06A51;
                v88 = sub_8D4620(v112[0]);
                v71 = sub_8D4620(*v8);
                sub_72BAF0(v109, v71 + v88, v70);
                sub_72BAF0(v110, -1, unk_4F06A60);
                if ( v5 )
                {
                  v72 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v5 + 64LL) + 16LL);
                  if ( !*(_QWORD *)(v72 + 16) )
                    goto LABEL_138;
                  v89 = *(_QWORD *)(v72 + 16);
                  sub_6F8800(v89, v5, v116);
                  v73 = sub_6F6F40(v116, 0);
                  v74 = v89;
                  v75 = v73;
                }
                else
                {
                  v75 = v95;
                  v74 = 0;
                }
                if ( !v75 )
                {
LABEL_138:
                  sub_724E30(&v109);
                  sub_724E30(&v110);
                  goto LABEL_9;
                }
                v90 = v7;
                v76 = v74;
                while ( 1 )
                {
                  if ( *(_BYTE *)(v75 + 24)
                    && *(_BYTE *)(*(_QWORD *)(v75 + 56) + 173LL) == 1
                    && (v82 = *(_QWORD *)(v75 + 56), (unsigned int)sub_621060(v82, v109) != -1)
                    && (unsigned int)sub_621060(v82, v110) )
                  {
                    sub_6851C0(0xA74u, (_DWORD *)(v82 + 64));
                    v102 = sub_72C930(2676);
                    if ( v5 )
                    {
                      v5[56] = 1;
LABEL_134:
                      v76 = *(_QWORD *)(v76 + 16);
                      if ( !v76 )
                        goto LABEL_137;
                      sub_6F8800(v76, v5, v116);
                      v75 = sub_6F6F40(v116, 0);
                      goto LABEL_136;
                    }
                  }
                  else if ( v5 )
                  {
                    goto LABEL_134;
                  }
                  v75 = *(_QWORD *)(v75 + 16);
LABEL_136:
                  if ( !v75 )
                  {
LABEL_137:
                    v7 = v90;
                    goto LABEL_138;
                  }
                }
              }
            }
            if ( v5 )
            {
              v5[56] = 1;
            }
            else
            {
              v19 = v113;
              v17 = 2520;
              v105 = v9;
              sub_6861A0(0x9D8u, v113, v112[0], *v8);
              LODWORD(v9) = v105;
            }
            goto LABEL_40;
          }
        }
        goto LABEL_41;
      }
    }
    else if ( v10 )
    {
      goto LABEL_41;
    }
    if ( !v108 && !v102 )
    {
LABEL_40:
      v86 = v9;
      v23 = sub_72C930(v17);
      LODWORD(v9) = v86;
      v102 = v23;
    }
LABEL_41:
    if ( !v100 )
      goto LABEL_50;
    goto LABEL_42;
  }
LABEL_4:
  if ( v107 )
  {
    LODWORD(v9) = 0;
  }
  else
  {
    v63 = sub_72C930(a1);
    LODWORD(v9) = 0;
    v102 = v63;
  }
  if ( v8 )
    goto LABEL_26;
LABEL_7:
  v10 = 0;
  if ( v100 )
  {
    sub_72C930(a1);
    goto LABEL_9;
  }
LABEL_50:
  v87 = v9;
  v100 = sub_688510((__int64)v5, v7, &v110);
  if ( !v100 )
  {
    if ( !(_DWORD)v110 )
    {
      if ( v102 )
        goto LABEL_46;
      v102 = sub_72C930(v5);
    }
    goto LABEL_9;
  }
  for ( j = *v7; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v29 = *(_QWORD *)(j + 160);
  v30 = sub_8D2780(v29);
  v31 = v87;
  if ( !v30 )
  {
    if ( v5 )
    {
      v5[56] = 1;
    }
    else
    {
      v29 = 2521;
      sub_685360(0x9D9u, (_DWORD *)v7 + 17, *v7);
      v31 = v87;
    }
    if ( v102 )
      goto LABEL_46;
    v101 = v31;
    v32 = sub_72C930(v29);
    v31 = v101;
    v102 = v32;
  }
  v100 = 0;
  if ( !v31 || v8 && !v10 )
  {
LABEL_9:
    if ( !v102 )
      goto LABEL_10;
LABEL_46:
    sub_6E6260(v4);
    goto LABEL_47;
  }
  if ( v102 )
    goto LABEL_46;
  v33 = sub_8D4620(v112[0]);
  v34 = *v7;
  if ( v33 != sub_8D4620(*v7) )
  {
    if ( v5 )
    {
      v5[56] = 1;
    }
    else
    {
      v34 = 2522;
      sub_6861A0(0x9DAu, (_DWORD *)v7 + 17, *v7, v112[0]);
    }
    v100 = 0;
    v102 = sub_72C930(v34);
    goto LABEL_9;
  }
  v100 = 0;
LABEL_10:
  if ( v107 )
  {
LABEL_49:
    sub_6F40C0(v112);
    v103 = *(_QWORD *)&dword_4D03B80;
    goto LABEL_12;
  }
LABEL_11:
  sub_6F69D0(v112, 0);
  v103 = 0;
LABEL_12:
  v11 = sub_6F6F40(v112, 0);
  v12 = (__int64 *)(v11 + 16);
  if ( v8 )
  {
    if ( v108 )
    {
      sub_6F40C0(v8);
      v103 = *(_QWORD *)&dword_4D03B80;
    }
    else
    {
      sub_6F69D0(v8, 0);
    }
    v13 = sub_6F6F40(v8, 0);
    *(_QWORD *)(v11 + 16) = v13;
    v12 = (__int64 *)(v13 + 16);
  }
  if ( v100 )
  {
    *v12 = v95;
    if ( v103 )
    {
      v14 = sub_726700(23);
      *(_QWORD *)v14 = v103;
      v15 = 58;
    }
    else
    {
      v59 = v112[0];
      if ( v92 )
      {
        while ( *(_BYTE *)(v59 + 140) == 12 )
          v59 = *(_QWORD *)(v59 + 160);
        v59 = sub_72B5A0(*(_QWORD *)(v59 + 160), v92, 0);
      }
      v60 = (__int64 *)sub_726700(23);
      *v60 = v59;
      v14 = (__int64)v60;
      v15 = 58;
    }
  }
  else
  {
    if ( (_DWORD)v110 )
    {
      sub_6F40C0(v7);
      v103 = *(_QWORD *)&dword_4D03B80;
    }
    else
    {
      sub_6F69D0(v7, 0);
    }
    *v12 = sub_6F6F40(v7, 0);
    if ( v103 )
    {
      v14 = sub_726700(23);
      *(_QWORD *)v14 = v103;
    }
    else
    {
      v64 = v112[0];
      v65 = (__int64 *)sub_726700(23);
      *v65 = v64;
      v14 = (__int64)v65;
    }
    v15 = 53;
  }
  *(_BYTE *)(v14 + 56) = v15;
  *(_QWORD *)(v14 + 64) = v11;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && *(_BYTE *)(v14 + 24) )
  {
    sub_6E70E0(v14, v116);
    LODWORD(v117) = v111;
    WORD2(v117) = WORD2(v111);
    *(_QWORD *)dword_4F07508 = v117;
    LODWORD(v118) = v96;
    WORD2(v118) = v98;
    *(_QWORD *)&dword_4F061D8 = v118;
    sub_6E3280(v116, &dword_4F077C8);
    sub_6F6F40(v116, 0);
  }
  sub_6E70E0(v14, v4);
LABEL_47:
  *(_DWORD *)(v4 + 68) = v111;
  *(_WORD *)(v4 + 72) = WORD2(v111);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v4 + 68);
  *(_DWORD *)(v4 + 76) = v96;
  *(_WORD *)(v4 + 80) = v98;
  *(_QWORD *)&dword_4F061D8 = *(_QWORD *)(v4 + 76);
  return sub_6E3280(v4, &v111);
}
