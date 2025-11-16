// Function: sub_852E40
// Address: 0x852e40
//
_DWORD *__fastcall sub_852E40(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  unsigned int v7; // r15d
  unsigned __int16 v8; // r14
  __int64 v9; // rax
  _QWORD *v10; // rdi
  int v11; // ecx
  const char *v12; // rsi
  unsigned int v13; // edx
  __int64 v14; // rcx
  _DWORD *result; // rax
  unsigned __int64 v16; // rax
  signed __int64 v17; // rax
  _BOOL8 v18; // r13
  int v19; // r14d
  int v20; // r15d
  unsigned __int8 *i; // rdi
  __int64 v22; // r9
  signed __int64 v23; // rax
  size_t v24; // rax
  __int64 v25; // r8
  char *v26; // rdi
  unsigned int v27; // r9d
  int v28; // edi
  unsigned __int8 *v29; // rax
  char *v30; // r12
  __int64 v31; // rdi
  char v32; // dl
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rbx
  unsigned __int8 *v38; // rax
  unsigned __int8 *v39; // r12
  unsigned __int8 *v40; // r12
  size_t v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  char *v46; // rax
  __int64 v47; // rdi
  char v48; // dl
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // r12
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  size_t v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // rax
  int v61; // eax
  __int64 v62; // rax
  __int64 v63; // r13
  __int64 v64; // rax
  __int64 v65; // r12
  _QWORD *v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  size_t v72; // r15
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  signed int v77; // r15d
  unsigned __int64 v78; // r12
  __int64 v79; // rax
  __int64 v80; // r13
  __int64 v81; // r15
  __int64 v82; // r15
  __off_t v83; // r12
  __int64 v84; // rsi
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  int v89; // r12d
  __int64 v90; // rax
  __int64 v91; // rdi
  FILE *v92; // rdi
  __int64 v93; // r13
  char *v94; // rax
  __int64 v95; // rax
  size_t v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  char *v101; // rax
  __int64 v102; // rax
  unsigned __int16 v103; // [rsp+12h] [rbp-4Eh]
  __int16 v104; // [rsp+14h] [rbp-4Ch]
  int v105; // [rsp+14h] [rbp-4Ch]
  unsigned int v106; // [rsp+14h] [rbp-4Ch]
  int v107; // [rsp+18h] [rbp-48h]
  unsigned int v108; // [rsp+18h] [rbp-48h]
  __int64 v109; // [rsp+18h] [rbp-48h]
  _BYTE ptr[56]; // [rsp+28h] [rbp-38h] BYREF

  v6 = dword_4F07508[0];
  v7 = dword_4F063F8;
  v8 = word_4F063FC[0];
  v107 = unk_4D03D20;
  v104 = dword_4F07508[1];
  dword_4D03CB0[0] = 1;
  unk_4D03D20 = 1;
  if ( unk_4F076D8 || qword_4F076D0 )
    sub_858B00();
  sub_7B8B50(a1, a2, a3, a4, a5, a6);
  if ( qword_4F064B0 )
    sub_7B2450();
  v9 = qword_4F5FB70;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  if ( !qword_4F5FB70 )
    goto LABEL_38;
  while ( 1 )
  {
    while ( *(_DWORD *)(v9 + 8) != 2 )
    {
LABEL_7:
      v9 = *(_QWORD *)v9;
      if ( !v9 )
        goto LABEL_11;
    }
    v13 = *(_DWORD *)(v9 + 12);
    if ( v13 > 2 )
      break;
    *(_QWORD *)(v9 + 40) = v12;
    v12 = (const char *)v9;
    v9 = *(_QWORD *)v9;
    ++v11;
    if ( !v9 )
      goto LABEL_11;
  }
  if ( v13 != 7 )
  {
LABEL_27:
    if ( !v11 && (v13 - 7 <= 2 || v13 == 13) )
      v10 = (_QWORD *)v9;
    goto LABEL_7;
  }
  if ( v11 )
  {
    --v11;
    if ( v12 )
      v12 = (const char *)*((_QWORD *)v12 + 5);
    goto LABEL_27;
  }
  dword_4D03CAC = 1;
LABEL_11:
  if ( v10 )
  {
    v14 = v10[4];
    *v10 = 0;
    qword_4F5FB68 = (__int64)v10;
    *(_QWORD *)&dword_4D03CA0 = v14;
    goto LABEL_13;
  }
LABEL_38:
  dword_4D03CAC = 1;
LABEL_13:
  sub_7CA670();
  dword_4F07508[0] = v6;
  dword_4F063F8 = v7;
  dword_4D03CB0[0] = 0;
  word_4F063FC[0] = v8;
  LOWORD(dword_4F07508[1]) = v104;
  unk_4D03D20 = v107;
  result = &dword_4D03CAC;
  if ( dword_4D03CAC )
    return result;
  if ( dword_4D044E8[0] )
  {
    v12 = ".pch";
    v105 = 0;
    LODWORD(v18) = 0;
    v19 = 1;
    v20 = 1;
    v108 = dword_4F077C8;
    v103 = word_4F077CC[0];
    for ( i = (unsigned __int8 *)sub_722560(qword_4F076F0, ".pch");
          i;
          i = (unsigned __int8 *)sub_722040(v28, qword_4D044D8, ".pch") )
    {
      if ( !v18 )
      {
        v29 = sub_851FD0(i, (__int64)v12);
        v30 = (char *)v29;
        v31 = (__int64)v29;
        v32 = dword_4D044E8[0];
        if ( dword_4D044E8[0] )
          v32 = 15;
        qword_4F5FB48 = (FILE *)sub_685EB0((__int64)v29, 1, v32, 1698);
        if ( qword_4F5FB48 )
        {
          qword_4D044F8 = v30;
          v37 = sub_852210(v31, 1, v33, v34, v35, v36);
          if ( qword_4F5FB48 )
          {
            fclose(qword_4F5FB48);
            qword_4F5FB48 = 0;
          }
          if ( v37 )
          {
            v22 = *(unsigned int *)(v37 + 32);
            v23 = v22 - v108;
            if ( (_DWORD)v22 == v108 )
              v23 = *(unsigned __int16 *)(v37 + 36) - (unsigned __int64)v103;
            v106 = *(_DWORD *)(v37 + 32);
            if ( v23 >= 0 )
            {
              v24 = strlen(v30) + 1;
              v26 = qword_4F5F840;
              v27 = v106;
              v103 = *(_WORD *)(v37 + 36);
              if ( qword_4F5F848 < v24 )
              {
                v93 = qword_4F5F848 + 1024;
                if ( v24 >= qword_4F5F848 + 1024 )
                  v93 = v24;
                v94 = (char *)sub_822C60(qword_4F5F840, qword_4F5F848, v93, *(unsigned __int16 *)(v37 + 36), v25, v106);
                qword_4F5F848 = v93;
                v27 = v106;
                qword_4F5F840 = v94;
                v26 = v94;
              }
              v12 = v30;
              v108 = v27;
              strcpy(v26, v30);
              v18 = qword_4F5FB68 == v37;
              if ( v20 )
                goto LABEL_118;
            }
            v105 = 1;
          }
          else if ( unk_4D044E0 )
          {
            v95 = sub_723260(v30);
            sub_684B10(dword_4F5F928, &dword_4F077C8, v95);
          }
          else
          {
            LODWORD(v18) = 0;
          }
        }
      }
      v28 = v19;
      v20 = 0;
      v19 = 0;
      v12 = qword_4D044D8;
    }
    if ( v105 )
    {
LABEL_118:
      v96 = strlen(qword_4F5F840);
      v101 = (char *)sub_822B10(v96 + 1, (__int64)v12, v97, v98, v99, v100);
      v12 = qword_4F5F840;
      qword_4D044F8 = v101;
      strcpy(v101, qword_4F5F840);
      if ( !dword_4D04500 )
      {
        if ( !dword_4D044E8[0] )
          goto LABEL_16;
LABEL_117:
        v47 = (__int64)qword_4D044F8;
        v48 = 15;
        goto LABEL_67;
      }
    }
    else if ( !dword_4D04500 )
    {
      goto LABEL_59;
    }
    if ( !dword_4D044E8[0] )
      goto LABEL_65;
    goto LABEL_117;
  }
  if ( !dword_4D04500 )
    goto LABEL_16;
LABEL_65:
  v40 = sub_851FD0((unsigned __int8 *)qword_4D044F8, (__int64)v12);
  qword_4D044F8 = (char *)v40;
  v41 = strlen((const char *)v40);
  v46 = (char *)sub_822B10(v41 + 1, (__int64)v12, v42, v43, v44, v45);
  qword_4D044F8 = strcpy(v46, (const char *)v40);
  v47 = (__int64)qword_4D044F8;
  v48 = dword_4D044E8[0];
  if ( dword_4D044E8[0] )
    v48 = 15;
LABEL_67:
  qword_4F5FB48 = (FILE *)sub_685EB0(v47, 1, v48, 1698);
  if ( !qword_4F5FB48 )
  {
LABEL_81:
    v63 = 0;
    v64 = sub_723260(qword_4D044F8);
    sub_67EA10(632, v64);
    dword_4D03C90 = 1;
    if ( dword_4F5F92C <= 0 )
    {
LABEL_88:
      v67 = qword_4F061C8;
      qword_4F061C8 = 0;
      v109 = v67;
      if ( fread(&qword_4F5F880, 0xA8u, 1u, qword_4F5FB48) == 1 && fread(&dword_4F073A8, 4u, 1u, qword_4F5FB48) == 1 )
      {
        sub_823100(dword_4F073A8, 4, v68, v69, v70, v71);
        v72 = 8LL * (dword_4F073A8 + 1);
        if ( fread(qword_4F073B0, v72, 1u, qword_4F5FB48) == 1
          && fread(qword_4F072B0, v72, 1u, qword_4F5FB48) == 1
          && fread(&dword_4F073A0, 4u, 1u, qword_4F5FB48) == 1 )
        {
          v77 = dword_4F073A0 + 1;
          if ( (int)(dword_4F073A0 + 1) <= 1
            || (sub_823040(dword_4F073A0, 4, v73, v74, v75, v76),
                fread(qword_4F072B8, 16LL * v77, 1u, qword_4F5FB48) == 1) )
          {
            sub_822A90();
            v78 = ftell(qword_4F5FB48);
            v79 = qword_4F5F868;
            if ( qword_4F5F868 > 0 )
            {
              v80 = 0;
              do
              {
                v81 = v80++;
                v82 = qword_4F5F870 + 16 * v81;
                v83 = sub_721A20(v78);
                sub_7218F0(qword_4F5FB48, 0, 0, v83, *(_QWORD *)(v82 + 8), *(void **)v82, (__int64)qword_4D044F8);
                v84 = *(_QWORD *)(v82 + 8);
                v78 = v84 + v83;
                sub_823270(*(_QWORD *)v82, v84, v85, v86, v87, v88);
                v79 = qword_4F5F868;
              }
              while ( qword_4F5F868 > v80 );
            }
            if ( qword_4F5F870 )
              sub_822B90(qword_4F5F870, 16 * v79);
            dword_4F5F878 = unk_4F06468;
            qword_4F061C8 = v109;
            sub_738350();
            qword_4F07280 = qword_4F5F880;
            qword_4F07288 = qword_4F5F888;
            unk_4F07290 = qword_4F5F890;
            qword_4F07308 = qword_4F5F908;
            unk_4F07310 = qword_4F5F910;
            qword_4F072C0 = qword_4F5F8C0;
            qword_4F07320[0] = qword_4F5F920;
            v89 = dword_4F04C30;
            dword_4F04C30 = 0;
            while ( v89 > dword_4F04C30 )
              sub_880E90();
            *(_QWORD *)(unk_4D03FF8 + 8LL) = qword_4D03FF0;
            sub_8D08E0();
            v90 = qword_4F061C8;
            v91 = qword_4F061C8 + 16LL;
            *(_QWORD *)(qword_4F061C8 + 8LL) = 0;
            *(_QWORD *)(v90 + 358) = 0;
            memset(
              (void *)(v91 & 0xFFFFFFFFFFFFFFF8LL),
              0,
              8 * ((unsigned __int64)((unsigned int)v90 - (v91 & 0xFFFFFFF8) + 366) >> 3));
            qword_4F07670 = 0;
            sub_7CA660();
            sub_738240();
            sub_825100();
            sub_823F90();
LABEL_103:
            v92 = qword_4F5FB48;
            if ( qword_4F5FB48 )
              goto LABEL_104;
            goto LABEL_59;
          }
        }
      }
    }
    else
    {
      while ( 1 )
      {
        v65 = qword_4F5F940[v63];
        v66 = *(_QWORD **)v65;
        if ( *(_QWORD *)v65 )
          break;
LABEL_87:
        if ( dword_4F5F92C <= (int)++v63 )
          goto LABEL_88;
      }
      while ( 1 )
      {
        if ( *(_BYTE *)(v65 + 16) )
          v66 = (_QWORD *)*v66;
        if ( fread(v66, *(_QWORD *)(v65 + 8), 1u, qword_4F5FB48) != 1 )
          break;
        v66 = *(_QWORD **)(v65 + 24);
        v65 += 24;
        if ( !v66 )
          goto LABEL_87;
      }
    }
LABEL_122:
    sub_851ED0();
  }
  v53 = sub_852210(v47, 1, v49, v50, v51, v52);
  if ( !v53 )
  {
    v61 = dword_4D044E8[0];
    goto LABEL_110;
  }
  if ( fread(ptr, 8u, 1u, qword_4F5FB48) != 1 )
    goto LABEL_122;
  if ( fread(&qword_4F5F868, 8u, 1u, qword_4F5FB48) != 1 )
    goto LABEL_122;
  v58 = 16 * qword_4F5F868;
  qword_4F5F870 = sub_822B10(16 * qword_4F5F868, 8, v54, v55, v56, v57);
  if ( fread((void *)qword_4F5F870, v58, 1u, qword_4F5FB48) != 1 )
    goto LABEL_122;
  if ( (__int64)qword_4F07388 <= 0 )
  {
LABEL_80:
    unk_4D03C88 = *(_QWORD *)(v53 + 32);
    goto LABEL_81;
  }
  v59 = 0;
  v60 = 0;
  while ( *(_QWORD *)((char *)qword_4F07380 + v60) == *(_QWORD *)(qword_4F5F870 + v60)
       && *(_QWORD *)((char *)qword_4F07380 + v60 + 8) == *(_QWORD *)(qword_4F5F870 + v60 + 8) )
  {
    ++v59;
    v60 += 16;
    if ( v59 == qword_4F07388 )
      goto LABEL_80;
  }
  dword_4F5F928 = 634;
  v61 = dword_4D044E8[0];
  if ( !dword_4D044E8[0] )
    goto LABEL_121;
  if ( unk_4D044E0 )
  {
    v62 = sub_723260(qword_4D044F8);
    sub_684B10(dword_4F5F928, &dword_4F077C8, v62);
    v61 = dword_4D044E8[0];
  }
LABEL_110:
  if ( !v61 )
  {
LABEL_121:
    v102 = sub_723260(qword_4D044F8);
    sub_684B10(dword_4F5F928, &dword_4F077C8, v102);
    goto LABEL_103;
  }
  v92 = qword_4F5FB48;
  if ( !qword_4F5FB48 )
    goto LABEL_60;
LABEL_104:
  fclose(v92);
  qword_4F5FB48 = 0;
LABEL_59:
  if ( dword_4D044E8[0] )
  {
LABEL_60:
    v38 = (unsigned __int8 *)sub_722560(qword_4F076F0, ".pch");
    v39 = sub_851FD0(v38, (__int64)".pch");
    if ( (unsigned int)sub_7244C0((__int64)v39) && (!dword_4D03C90 || sub_722E50((char *)v39, qword_4D044F8, 0, 0, 0)) )
      sub_7212E0((__int64)v39);
    if ( !dword_4D044E8[0] )
      goto LABEL_16;
LABEL_17:
    v16 = dword_4D03CA0 - (unsigned __int64)(unsigned int)dword_4F077C8;
    if ( dword_4D03CA0 == dword_4F077C8 )
      v16 = unk_4D03CA4 - (unsigned __int64)word_4F077CC[0];
    if ( v16 )
    {
      if ( !dword_4D03C90 )
        goto LABEL_24;
      v17 = dword_4D03CA0 - (unsigned __int64)unk_4D03C88;
      if ( dword_4D03CA0 == unk_4D03C88 )
        v17 = unk_4D03CA4 - (unsigned __int64)unk_4D03C8C;
      if ( v17 > 0 )
      {
LABEL_24:
        result = dword_4D03C98;
        dword_4D03C98[0] = 1;
        return result;
      }
    }
  }
  else
  {
LABEL_16:
    if ( dword_4D04504 )
      goto LABEL_17;
  }
  result = dword_4D03C98;
  if ( !dword_4D03C98[0] )
    return sub_852D60();
  return result;
}
