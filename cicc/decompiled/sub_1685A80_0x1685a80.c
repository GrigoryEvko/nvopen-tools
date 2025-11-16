// Function: sub_1685A80
// Address: 0x1685a80
//
int __fastcall sub_1685A80(unsigned int *a1, __int64 a2, __int64 a3)
{
  void *v3; // r14
  _DWORD *v4; // r12
  _DWORD *v5; // rax
  __int64 v6; // r13
  _BYTE *v7; // rax
  const char *v8; // r12
  __int64 v9; // rdx
  int v10; // ecx
  int v11; // r8d
  int v12; // r9d
  int v13; // eax
  int v14; // edx
  int v15; // ecx
  int v16; // r8d
  int v17; // r9d
  int v18; // edx
  int v19; // ecx
  int v20; // r8d
  int v21; // r9d
  int v22; // edx
  int v23; // ecx
  int v24; // r8d
  int v25; // r9d
  int v26; // edx
  int v27; // ecx
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rax
  struct __jmp_buf_tag *v33; // rdi
  __int64 v34; // r15
  __int64 v35; // rsi
  __int64 v36; // r15
  __int64 v37; // rdx
  const char *v38; // rsi
  const char *v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rdx
  int v42; // eax
  int v43; // ecx
  int v44; // r8d
  int v45; // r9d
  int v46; // edx
  int v47; // ecx
  int v48; // r8d
  int v49; // r9d
  int v50; // r8d
  int v51; // r9d
  int v52; // ecx
  int v53; // r14d
  __int64 v54; // rsi
  __int64 v55; // rdi
  int v56; // edx
  int v57; // r8d
  int v58; // r9d
  __int64 v59; // rdx
  char *i; // r14
  char v61; // bl
  int j; // r12d
  __int64 v63; // rax
  void *v64; // rdx
  __int64 v65; // rsi
  __int64 v66; // rdi
  __int64 v67; // rdx
  char v68; // al
  __int64 v69; // rdx
  int v70; // eax
  int v71; // ecx
  int v72; // r8d
  int v73; // r9d
  const char *v74; // rdi
  __int64 v75; // rsi
  __int64 v76; // rdx
  int v77; // r9d
  _QWORD *v78; // r15
  __int64 v79; // rcx
  const char *v80; // r8
  __int64 v81; // rax
  int v82; // edx
  int v83; // ecx
  _QWORD *v84; // rdi
  int v85; // r8d
  int v86; // r9d
  __int64 v87; // rdx
  __int64 v88; // rax
  void *v89; // rdx
  __int64 v90; // r15
  __int64 v91; // rsi
  __int64 v92; // rdx
  char v93; // al
  __int64 v94; // rax
  void *v95; // rdx
  __int64 v96; // r15
  __int64 v97; // rsi
  __int64 v98; // rdx
  char v99; // al
  __int64 v100; // r14
  __int64 v101; // rbx
  __int64 v102; // rdi
  char *v103; // rdi
  FILE *v104; // rax
  _IO_FILE *v105; // r14
  int v106; // eax
  __int64 v107; // rax
  int v108; // eax
  unsigned int v109; // eax
  __int64 v110; // r14
  unsigned int v111; // r13d
  _IO_FILE *v112; // rbx
  int v113; // r8d
  __int64 v114; // rax
  __int64 v115; // r12
  int v116; // eax
  __int64 v118; // rdi
  int v119; // edx
  __int64 v120; // rax
  _DWORD *v122; // [rsp+0h] [rbp-70h]
  int v123; // [rsp+8h] [rbp-68h]
  _DWORD *v124; // [rsp+8h] [rbp-68h]
  char *v125; // [rsp+10h] [rbp-60h]
  char *s; // [rsp+18h] [rbp-58h]
  unsigned int sb; // [rsp+18h] [rbp-58h]
  int sa; // [rsp+18h] [rbp-58h]
  unsigned int v129; // [rsp+24h] [rbp-4Ch]
  unsigned int v130; // [rsp+28h] [rbp-48h]
  char *v131; // [rsp+28h] [rbp-48h]
  unsigned int v132; // [rsp+28h] [rbp-48h]
  unsigned int v133; // [rsp+28h] [rbp-48h]
  int v134; // [rsp+28h] [rbp-48h]
  int v135; // [rsp+30h] [rbp-40h]

  v3 = (void *)a3;
  v4 = (_DWORD *)a2;
  if ( a2 && (v5 = *(_DWORD **)(*(_QWORD *)a2 + 48LL)) != 0 )
  {
    a3 = (*v5 >> 1) & 1;
    v135 = *v5 & 1;
    v130 = (*v5 >> 1) & 1;
    v129 = (*v5 >> 2) & 1;
  }
  else
  {
    v135 = 0;
    v130 = 1;
    v129 = 0;
  }
  v6 = *a1;
  v7 = &off_4984F18;
  if ( (_DWORD)v6 == 3 )
  {
    LODWORD(v7) = sub_1685A20((__int64)a1, a2, a3);
    if ( !(_BYTE)v7 )
    {
      if ( (unsigned __int8)sub_1685A10((__int64)a1, a2, a3) )
      {
        if ( a1 != (unsigned int *)&unk_4CD28D0 )
        {
          v63 = sub_1688290(128);
          v64 = v3;
          v39 = "error   ";
          v6 = v63;
          v65 = *((_QWORD *)a1 + 1);
          sub_1688540(v63, v65, v64);
          v66 = v6;
          LODWORD(v6) = 5;
          v125 = (char *)sub_16884C0(v66);
          v36 = sub_1688290(128);
          v68 = sub_1685A60(128, v65, v67);
          v38 = byte_3F871B3;
          if ( v68 )
            v38 = "@E@";
          goto LABEL_18;
        }
        v8 = "error   ";
      }
      else
      {
        if ( a1 != (unsigned int *)&unk_4CD28D0 )
        {
          v94 = sub_1688290(128);
          v95 = v3;
          v39 = "warning ";
          v96 = v94;
          v97 = *((_QWORD *)a1 + 1);
          sub_1688540(v94, v97, v95);
          v125 = (char *)sub_16884C0(v96);
          v36 = sub_1688290(128);
          v99 = sub_1685A60(128, v97, v98);
          v38 = byte_3F871B3;
          if ( v99 )
            v38 = "@W@";
          goto LABEL_18;
        }
        v8 = "warning ";
      }
LABEL_12:
      if ( sub_1685A40((__int64)a1, a2, a3) )
      {
        v13 = sub_1685A40((__int64)a1, a2, v9);
        sub_1689310(v13, a2, v14, v15, v16, v17);
        sub_1689310((unsigned int)" ", a2, v18, v19, v20, v21);
      }
      sub_1689310((unsigned int)"%s%s", (unsigned int)byte_3F871B3, (_DWORD)v8, v10, v11, v12);
      sub_1689310((unsigned int)": ", (unsigned int)byte_3F871B3, v22, v23, v24, v25);
      sub_16892D0(*((char **)a1 + 1), v3);
      sub_1689310((unsigned int)"\n", (_DWORD)v3, v26, v27, v28, v29);
      *(_BYTE *)(sub_1689050("\n", v3, v30) + 1) = 1;
      v32 = sub_1689050("\n", v3, v31);
      v33 = *(struct __jmp_buf_tag **)(v32 + 8);
      if ( !v33 )
LABEL_57:
        sub_16895D0();
LABEL_15:
      *(_QWORD *)(v32 + 16) = a1;
      longjmp(v33, 1);
    }
LABEL_11:
    LODWORD(v8) = 0;
    if ( a1 != (unsigned int *)&unk_4CD28D0 )
      return (int)v7;
    goto LABEL_12;
  }
  if ( (_DWORD)v6 == 2 )
  {
    LODWORD(v7) = sub_1685A30((__int64)a1, a2, a3);
    if ( !(_BYTE)v7 )
    {
      if ( a1 != (unsigned int *)&unk_4CD28D0 )
      {
        v88 = sub_1688290(128);
        v89 = v3;
        v39 = "info    ";
        v90 = v88;
        v91 = *((_QWORD *)a1 + 1);
        sub_1688540(v88, v91, v89);
        v125 = (char *)sub_16884C0(v90);
        v36 = sub_1688290(128);
        v93 = sub_1685A60(128, v91, v92);
        v38 = byte_3F871B3;
        if ( v93 )
          v38 = "@O@";
        goto LABEL_18;
      }
      v8 = "info    ";
      goto LABEL_12;
    }
    goto LABEL_11;
  }
  if ( a1 == (unsigned int *)&unk_4CD28D0 )
  {
    v8 = (const char *)off_4984EE0[v6];
    goto LABEL_12;
  }
  if ( !(_DWORD)v6 )
    return (int)v7;
  v34 = sub_1688290(128);
  v35 = *((_QWORD *)a1 + 1);
  sub_1688540(v34, v35, v3);
  v125 = (char *)sub_16884C0(v34);
  v36 = sub_1688290(128);
  if ( (unsigned __int8)sub_1685A60(128, v35, v37) )
  {
    switch ( (int)v6 )
    {
      case 1:
        v39 = byte_3F871B3;
        v38 = "@I@";
        break;
      case 4:
        v39 = "error*  ";
        v38 = "@E@";
        break;
      case 5:
        v39 = "error   ";
        v38 = "@E@";
        break;
      case 6:
        v39 = "fatal   ";
        v38 = "@E@";
        break;
      default:
        v38 = byte_3F871B3;
        v39 = (const char *)off_4984EE0[(unsigned int)v6];
        break;
    }
  }
  else
  {
    v38 = byte_3F871B3;
    v39 = (const char *)off_4984EE0[(unsigned int)v6];
  }
LABEL_18:
  sub_16884F0(v36, v38);
  if ( sub_1685A40(v36, (__int64)v38, v40) )
  {
    v42 = sub_1685A40(v36, (__int64)v38, v41);
    sub_1688630(v36, (unsigned int)"%s", v42, v43, v44, v45);
    sub_1688630(v36, (unsigned int)" ", v46, v47, v48, v49);
  }
  s = (char *)sub_1688450(v36);
  if ( v4 )
  {
    if ( *(_QWORD *)v4 )
    {
      v52 = v4[2];
      if ( v52 != 0xFFFFFFF )
        sub_1688630(v36, (unsigned int)"%s, line %d; ", *(_QWORD *)(*(_QWORD *)v4 + 24LL), v52, v50, v51);
    }
  }
  sub_1688630(v36, (unsigned int)"%s%s", (unsigned int)byte_3F871B3, (_DWORD)v39, v50, v51);
  v53 = sub_16886C0(v36);
  v54 = (__int64)": ";
  v55 = v36;
  v123 = v53 - strlen(s);
  sub_1688630(v36, (unsigned int)": ", v56, v123, v57, v58);
  for ( i = v125; ; ++i )
  {
    v61 = *i;
    if ( !*i )
      break;
    v54 = (unsigned int)v61;
    v55 = v36;
    sub_1688520(v36, v54);
    if ( v61 == 10 && !(unsigned __int8)sub_1685A70(v36, v54, v59) )
    {
      sub_16884F0(v36, s);
      if ( v123 )
      {
        v122 = v4;
        for ( j = 0; j != v123; ++j )
          sub_1688520(v36, 32);
        v4 = v122;
      }
      v54 = (__int64)". ";
      v55 = v36;
      sub_16884F0(v36, ". ");
    }
  }
  if ( sub_1685A50(v55, v54, v59) )
  {
    v70 = sub_1685A50(v55, v54, v69);
    sub_1688630(v36, (unsigned int)" %s", v70, v71, v72, v73);
  }
  sub_1688520(v36, 10);
  sub_16856A0(s);
  v74 = (const char *)v36;
  v75 = v130;
  v78 = (_QWORD *)sub_16884C0(v36);
  if ( v130 )
  {
    v79 = v129;
    v80 = byte_3F871B3;
    if ( !v129 )
    {
LABEL_42:
      v74 = (const char *)qword_4F9F380[byte_42AE118[(unsigned int)v6]];
      if ( v74 )
      {
        v75 = (__int64)"%s%s";
        v131 = (char *)v80;
        sub_1688A60((_DWORD)v74, (unsigned int)"%s%s", (_DWORD)v80, (_DWORD)v78, (_DWORD)v80, v77, (char)v122);
      }
      else
      {
        v75 = (__int64)v80;
        v74 = "%s%s";
        v131 = (char *)v80;
        sub_1689540((unsigned int)"%s%s", (_DWORD)v80, (_DWORD)v78, v79, (_DWORD)v80, v77, (char)v122);
      }
      v76 = v129;
      if ( v129 )
      {
        v74 = v131;
        sub_16856A0(v131);
      }
      goto LABEL_46;
    }
    v100 = *(_QWORD *)v4;
    v101 = *(_QWORD *)(*(_QWORD *)v4 + 48LL);
    if ( *(_QWORD *)v4 != *(_QWORD *)(v101 + 32) )
    {
      v102 = *(_QWORD *)(v101 + 40);
      if ( v102 )
      {
        sub_1684100(v102);
        fclose(*(FILE **)(v101 + 48));
      }
      *(_QWORD *)(v101 + 32) = v100;
      v103 = *(char **)(v100 + 24);
      v75 = (__int64)"r";
      v104 = fopen(v103, "r");
      v105 = v104;
      if ( !v104 )
      {
        *(_QWORD *)(v101 + 40) = 0;
LABEL_99:
        v118 = *(_QWORD *)(sub_1689050(v103, v75, v76) + 24);
        v80 = (const char *)sub_1685080(v118, 1);
        if ( !v80 )
        {
          sub_1683C30(v118, 1, v119, v79, 0, v77, (char)v122);
          v80 = 0;
        }
        *v80 = 0;
        goto LABEL_42;
      }
      v106 = _IO_getc(v104);
      *(_QWORD *)(v101 + 48) = v105;
      v75 = (__int64)sub_16881E0;
      sb = v106;
      v107 = sub_1684080(sub_16881D0, sub_16881E0, 0x400u);
      v79 = sb;
      v132 = 0;
      *(_QWORD *)(v101 + 40) = v107;
      v108 = sb;
LABEL_79:
      while ( v108 != -1 )
      {
        if ( v108 != 10 )
        {
          while ( 1 )
          {
            v108 = _IO_getc(v105);
            if ( v108 == 10 )
              break;
            if ( v108 == -1 )
              goto LABEL_79;
          }
        }
        ++v132;
        if ( __ROR4__(-858993459 * v132, 1) <= 0x19999999u )
        {
          v120 = ftell(v105);
          v75 = v132 / 0xAuLL;
          sub_1684190(*(_QWORD *)(v101 + 40), v75, v120);
        }
        v108 = _IO_getc(v105);
      }
    }
    v103 = *(char **)(v101 + 40);
    if ( v103 )
    {
      v133 = v4[2] - 1;
      v109 = sub_1684840((__int64)v103, v133 / 0xA, v76, v79);
      v103 = *(char **)(v101 + 48);
      v75 = v109;
      if ( !fseek((FILE *)v103, v109, 0) )
      {
        v124 = v4;
        v110 = v101;
        sa = v6;
        v111 = v133 % 0xA;
        do
        {
          v112 = *(_IO_FILE **)(v110 + 48);
          v103 = (char *)v112;
          v113 = feof(v112);
          v114 = 0;
          if ( !v113 )
          {
            v115 = sub_1688290(128);
            v134 = _IO_getc(v112);
            sub_16884F0(v115, "# ");
            LOBYTE(v116) = v134;
            if ( v134 != 10 && v134 != -1 )
            {
              do
              {
                sub_1688520(v115, (unsigned int)(char)v116);
                v116 = _IO_getc(v112);
              }
              while ( v116 != -1 && v116 != 10 );
            }
            v75 = 10;
            sub_1688520(v115, 10);
            v103 = (char *)v115;
            v114 = sub_16884C0(v115);
          }
        }
        while ( v111-- != 0 );
        LODWORD(v6) = sa;
        v4 = v124;
        v80 = (const char *)v114;
        if ( v114 )
          goto LABEL_42;
      }
    }
    goto LABEL_99;
  }
LABEL_46:
  if ( v135 )
  {
    v81 = sub_1689050(v74, v75, v76);
    v84 = sub_1685080(*(_QWORD *)(v81 + 24), 24);
    if ( !v84 )
    {
      sub_1683C30(0, 24, v82, v83, v85, v86, (char)v122);
      v84 = 0;
    }
    v84[2] = 0;
    *(_OWORD *)v84 = 0;
    *(_DWORD *)v84 = v4[2];
    v84[1] = a1;
    v84[2] = v125;
    v75 = *(_QWORD *)v4 + 40LL;
    LODWORD(v7) = (unsigned int)sub_1683B10((__int64)v84, (__int64 *)v75);
  }
  else
  {
    v84 = v125;
    LODWORD(v7) = sub_16856A0(v125);
  }
  if ( (unsigned int)v6 > 2 )
  {
    v7 = (_BYTE *)sub_1689050(v84, v75, v87);
    *v7 = 1;
    if ( (unsigned int)v6 > 4 )
    {
      v7 = (_BYTE *)sub_1689050(v84, v75, v87);
      v7[1] = 1;
    }
  }
  if ( v78 )
  {
    v84 = v78;
    LODWORD(v7) = sub_16856A0(v78);
  }
  if ( (unsigned int)v6 > 5 )
  {
    v32 = sub_1689050(v84, v75, v87);
    v33 = *(struct __jmp_buf_tag **)(v32 + 8);
    if ( !v33 )
      goto LABEL_57;
    goto LABEL_15;
  }
  return (int)v7;
}
