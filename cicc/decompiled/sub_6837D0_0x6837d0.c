// Function: sub_6837D0
// Address: 0x6837d0
//
__int64 __fastcall sub_6837D0(__int64 a1, FILE *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r12
  unsigned __int8 v4; // bl
  __int64 result; // rax
  __int64 v6; // rdx
  bool v7; // zf
  __int64 v8; // rdx
  _BYTE *v9; // rax
  int v10; // ebx
  unsigned int v11; // r13d
  __int64 v12; // rdi
  char v13; // al
  int v14; // r15d
  __int64 v15; // r12
  unsigned int v16; // eax
  bool v17; // cc
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  _QWORD *v22; // rdi
  __int64 v23; // rax
  size_t v24; // rax
  __int64 v25; // rdi
  unsigned __int8 v26; // al
  __int64 v27; // rbx
  __int64 v28; // rdi
  int v29; // r13d
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rax
  int v34; // r14d
  int v35; // r13d
  int v36; // ebx
  __int64 v37; // r15
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rcx
  _DWORD *v42; // rax
  _QWORD *v43; // r14
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  _DWORD *v47; // rax
  _QWORD *v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // r14
  _DWORD *v56; // rax
  _QWORD *v57; // rbx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  char v62; // r13
  _DWORD *v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // rax
  char *v71; // rax
  __int64 v72; // rsi
  const char *v73; // rdi
  __int64 v74; // rdx
  FILE *v75; // rcx
  char *v76; // rax
  int v77; // [rsp+8h] [rbp-A8h]
  int v78; // [rsp+Ch] [rbp-A4h]
  int v79; // [rsp+10h] [rbp-A0h]
  bool v80; // [rsp+17h] [rbp-99h]
  int v81; // [rsp+18h] [rbp-98h]
  int v82; // [rsp+1Ch] [rbp-94h]
  __int64 v83; // [rsp+20h] [rbp-90h]
  _QWORD *v84; // [rsp+20h] [rbp-90h]
  __int64 v85; // [rsp+38h] [rbp-78h]
  unsigned __int64 v86; // [rsp+48h] [rbp-68h]
  _BOOL4 v87; // [rsp+48h] [rbp-68h]
  int v88; // [rsp+5Ch] [rbp-54h] BYREF
  char s[80]; // [rsp+60h] [rbp-50h] BYREF

  v2 = 0;
  v3 = a1;
  dword_4CFDEAC = 0;
  v4 = byte_4F07481[0];
  result = *(unsigned __int8 *)(a1 + 180);
  if ( byte_4F07481[0] > (unsigned __int8)result )
    goto LABEL_2;
  a1 = *(unsigned int *)(a1 + 96);
  v7 = (unsigned int)sub_729F80(a1) == 0;
  result = *(unsigned __int8 *)(v3 + 180);
  if ( v7 )
  {
    if ( unk_4D042B8 )
      v4 = 7;
  }
  else
  {
    v4 = 8;
  }
  if ( v4 > (unsigned __int8)result )
  {
    v2 = 0;
    goto LABEL_2;
  }
  v8 = *(int *)(v3 + 176);
  if ( (unsigned __int8)result <= 7u )
  {
    v18 = byte_4CFFE80[4 * v8 + 2];
    if ( (v18 & 1) != 0 )
    {
      byte_4CFFE80[4 * v8 + 2] = v18 | 2;
      if ( (v18 & 2) != 0 )
      {
        if ( (v18 & 2) == 0 )
          goto LABEL_22;
        goto LABEL_50;
      }
    }
    else
    {
      byte_4CFFE80[4 * v8 + 2] = v18 | 2;
    }
  }
  else
  {
    byte_4CFFE80[4 * v8 + 2] |= 2u;
  }
  if ( dword_4F04C64 == -1 )
    goto LABEL_22;
  if ( !(unsigned int)sub_67D520(*(_DWORD *)(v3 + 176), *(_BYTE *)(v3 + 180), (unsigned int *)(v3 + 96)) )
  {
    if ( dword_4F04C44 != -1
      || (v9 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64), (v9[6] & 6) != 0)
      || v9[4] == 12
      || (v9[12] & 0xC) != 0 )
    {
      sub_67D470(*(_DWORD *)(v3 + 176), *(_BYTE *)(v3 + 180), (unsigned int *)(v3 + 96));
    }
LABEL_22:
    if ( unk_4D042B8 && !unk_4D042B0 && *(_DWORD *)(v3 + 176) != 992 )
      sub_684920(992);
    v10 = *(unsigned __int8 *)(v3 + 180);
    if ( (unsigned __int8)v10 <= 7u && (unsigned __int8)v10 >= unk_4F07480 )
    {
      v11 = 7;
      v10 = 7;
      if ( unk_4F04C48 == -1 )
      {
        sub_67B780(7, &qword_4F074A0);
        if ( !unk_4F07490 )
          goto LABEL_31;
LABEL_90:
        a1 = v11;
        sub_67B780(v11, &qword_4F074C0);
        a2 = (FILE *)qword_4F074E0;
        if ( qword_4F074E0 )
        {
          a1 = v11;
          sub_67B780(v11, qword_4F074E0);
        }
        result = *(unsigned __int8 *)(v3 + 180);
        v2 = 0;
        goto LABEL_2;
      }
    }
    else
    {
      v11 = (unsigned __int8)v10;
      if ( unk_4F04C48 == -1 || (unsigned __int8)v10 <= 6u )
        goto LABEL_29;
    }
    v19 = qword_4F04C68[0] + 776LL * unk_4F04C48;
    v20 = *(_QWORD *)(v19 + 368);
    if ( v20 )
    {
      v21 = *(_QWORD *)(v20 + 88);
      if ( !*(_QWORD *)(v21 + 48) )
        *(_QWORD *)(v21 + 48) = *(_QWORD *)(v19 + 360);
    }
LABEL_29:
    sub_67B780(v11, &qword_4F074A0);
    if ( !unk_4F07490 || (unsigned int)(v10 - 9) <= 2 )
    {
LABEL_31:
      if ( qword_4F074E0 )
        sub_67B780(v11, qword_4F074E0);
      sub_67B660();
      v12 = qword_4D039D8;
      if ( !qword_4D039D8 )
      {
        qword_4D039D8 = sub_8237A0(1024);
        v12 = qword_4D039D8;
      }
      sub_823800(v12);
      v13 = *(_BYTE *)(v3 + 180);
      if ( v13 == 9 )
      {
        if ( dword_4D03A08 )
        {
          v76 = sub_67C860(1510);
          fprintf(qword_4F07510, "%s\n", v76);
          sub_7235F0(9);
        }
        v7 = *(_DWORD *)(v3 + 176) == 3709;
        dword_4D03A08 = 1;
        if ( v7 || !unk_4F07464 )
        {
LABEL_44:
          sub_823800(qword_4D039E8);
          a1 = *(unsigned int *)(v3 + 176);
          a2 = (FILE *)*(unsigned __int8 *)(v3 + 180);
          if ( !(unsigned int)sub_67D310(a1, (unsigned __int8)a2) )
          {
            if ( !unk_4D04198 )
            {
              a1 = v3;
              sub_681D20(v3);
              v2 = 1;
              result = *(unsigned __int8 *)(v3 + 180);
              goto LABEL_2;
            }
            if ( unk_4D04198 == 1 )
            {
              v22 = (_QWORD *)qword_4D039D8;
              v23 = *(_QWORD *)(qword_4D039D8 + 16);
              if ( unk_4F074B0 + unk_4F074B8 > 1u )
              {
                if ( (unsigned __int64)(v23 + 1) > *(_QWORD *)(qword_4D039D8 + 8) )
                {
                  sub_823810(qword_4D039D8);
                  v22 = (_QWORD *)qword_4D039D8;
                  v23 = *(_QWORD *)(qword_4D039D8 + 16);
                }
                *(_BYTE *)(v22[4] + v23) = 44;
                v23 = v22[2] + 1LL;
                v22[2] = v23;
              }
              if ( (unsigned __int64)(v23 + 1) > v22[1] )
              {
                sub_823810(v22);
                v22 = (_QWORD *)qword_4D039D8;
                v23 = *(_QWORD *)(qword_4D039D8 + 16);
              }
              *(_BYTE *)(v22[4] + v23) = 123;
              ++v22[2];
              sub_8238B0(v22, "\"ruleId\":", 9);
              sub_8238B0(qword_4D039D8, "\"EC", 3);
              sprintf(s, "%lu", *(unsigned int *)(v3 + 176));
              v24 = strlen(s);
              sub_8238B0(qword_4D039D8, s, v24);
              sub_8238B0(qword_4D039D8, "\"", 1);
              v25 = qword_4D039D8;
              sub_8238B0(qword_4D039D8, ",\"level\":", 9);
              v26 = *(_BYTE *)(v3 + 180);
              if ( v26 > 7u || v26 < unk_4F07480 )
              {
                switch ( v26 )
                {
                  case 4u:
                    sub_8238B0(qword_4D039D8, "\"remark\"", 8);
                    goto LABEL_75;
                  case 5u:
                    sub_8238B0(qword_4D039D8, "\"warning\"", 9);
                    goto LABEL_75;
                  case 7u:
                  case 8u:
                    break;
                  case 9u:
                    sub_8238B0(qword_4D039D8, "\"catastrophe\"", 13);
                    goto LABEL_75;
                  case 0xBu:
                    sub_8238B0(qword_4D039D8, "\"internal_error\"", 16);
                    goto LABEL_75;
                  default:
                    sub_721090(v25);
                }
              }
              sub_8238B0(qword_4D039D8, "\"error\"", 7);
LABEL_75:
              sub_8238B0(qword_4D039D8, ",\"message\":", 11);
              sub_683690(v3);
              if ( *(_DWORD *)(v3 + 136) )
              {
                sub_8238B0(qword_4D039D8, ",\"locations\":", 13);
                sub_8238B0(qword_4D039D8, "[{\"physicalLocation\":", 21);
                sub_67C120((unsigned int *)(v3 + 136));
                sub_8238B0(qword_4D039D8, "}]", 2);
              }
              v27 = *(_QWORD *)(v3 + 72);
              v28 = qword_4D039D8;
              if ( v27 )
              {
                v29 = 1;
                sub_8238B0(qword_4D039D8, ",\"relatedLocations\":[", 21);
                v28 = qword_4D039D8;
                while ( 1 )
                {
                  *(_QWORD *)(v27 + 16) = v3;
                  sub_8238B0(v28, "{\"message\":", 11);
                  sub_683690(v27);
                  if ( *(_DWORD *)(v27 + 136) )
                  {
                    sub_8238B0(qword_4D039D8, ",\"physicalLocation\":", 20);
                    sub_67C120((unsigned int *)(v27 + 136));
                  }
                  v28 = qword_4D039D8;
                  v30 = *(_QWORD *)(qword_4D039D8 + 16);
                  if ( (unsigned __int64)(v30 + 1) > *(_QWORD *)(qword_4D039D8 + 8) )
                  {
                    sub_823810(qword_4D039D8);
                    v28 = qword_4D039D8;
                    v30 = *(_QWORD *)(qword_4D039D8 + 16);
                  }
                  *(_BYTE *)(*(_QWORD *)(v28 + 32) + v30) = 125;
                  v31 = *(_QWORD *)(v28 + 16);
                  v32 = v31 + 1;
                  *(_QWORD *)(v28 + 16) = v31 + 1;
                  if ( !v29 )
                  {
                    if ( (unsigned __int64)(v31 + 2) > *(_QWORD *)(v28 + 8) )
                    {
                      sub_823810(v28);
                      v28 = qword_4D039D8;
                      v32 = *(_QWORD *)(qword_4D039D8 + 16);
                    }
                    *(_BYTE *)(*(_QWORD *)(v28 + 32) + v32) = 44;
                    ++*(_QWORD *)(v28 + 16);
                  }
                  v27 = *(_QWORD *)(v27 + 8);
                  if ( !v27 )
                    break;
                  v29 = 0;
                }
                v67 = *(_QWORD *)(v28 + 16);
                if ( (unsigned __int64)(v67 + 1) > *(_QWORD *)(v28 + 8) )
                {
                  sub_823810(v28);
                  v28 = qword_4D039D8;
                  v67 = *(_QWORD *)(qword_4D039D8 + 16);
                }
                *(_BYTE *)(*(_QWORD *)(v28 + 32) + v67) = 93;
                v68 = *(_QWORD *)(v28 + 16) + 1LL;
                *(_QWORD *)(v28 + 16) = v68;
              }
              else
              {
                v68 = *(_QWORD *)(qword_4D039D8 + 16);
              }
              if ( (unsigned __int64)(v68 + 1) > *(_QWORD *)(v28 + 8) )
              {
                sub_823810(v28);
                v28 = qword_4D039D8;
                v68 = *(_QWORD *)(qword_4D039D8 + 16);
              }
              *(_BYTE *)(*(_QWORD *)(v28 + 32) + v68) = 125;
              v69 = *(_QWORD *)(v28 + 16);
              v70 = v69 + 1;
              *(_QWORD *)(v28 + 16) = v69 + 1;
              if ( (unsigned __int64)(v69 + 2) > *(_QWORD *)(v28 + 8) )
              {
                sub_823810(v28);
                v28 = qword_4D039D8;
                v70 = *(_QWORD *)(qword_4D039D8 + 16);
              }
              *(_BYTE *)(*(_QWORD *)(v28 + 32) + v70) = 0;
              a2 = qword_4F07510;
              ++*(_QWORD *)(v28 + 16);
              fputs(*(const char **)(v28 + 32), a2);
              a1 = (__int64)qword_4F07510;
              fflush(qword_4F07510);
            }
          }
          result = *(unsigned __int8 *)(v3 + 180);
          v2 = 1;
          goto LABEL_2;
        }
      }
      else if ( v13 == 11 || *(_DWORD *)(v3 + 176) == 3709 )
      {
        goto LABEL_44;
      }
      if ( dword_4F04C64 <= 0 )
      {
        if ( !dword_4F07588 )
          goto LABEL_44;
        if ( !qword_4D03FD0 )
          goto LABEL_44;
        if ( !*qword_4D03FD0 )
          goto LABEL_44;
        v14 = 0;
        v33 = sub_729AB0(*(unsigned int *)(v3 + 96));
        if ( !v33 )
          goto LABEL_44;
      }
      else
      {
        v85 = v3;
        v14 = 0;
        v15 = 776LL * dword_4F04C64;
        v86 = 776 * (dword_4F04C64 - (unsigned __int64)(unsigned int)(dword_4F04C64 - 1));
        while ( 1 )
        {
          v14 -= ((unsigned int)sub_67B7E0(v15 + qword_4F04C68[0], s, &v88, 0, 0) == 0) - 1;
          if ( v86 == v15 )
            break;
          v15 -= 776;
        }
        v3 = v85;
        if ( !dword_4F07588 )
          goto LABEL_43;
        if ( !qword_4D03FD0 )
          goto LABEL_43;
        if ( !*qword_4D03FD0 )
          goto LABEL_43;
        v33 = sub_729AB0(*(unsigned int *)(v85 + 96));
        if ( !v33 )
          goto LABEL_43;
      }
      if ( qword_4D03FD0[22] != v33 )
      {
        v78 = 1;
        ++v14;
LABEL_98:
        v80 = 0;
        if ( dword_4F07474 > 0 )
          v80 = dword_4F07474 + 1 < v14;
        v82 = dword_4F07474 / 2;
        if ( v14 != 1 )
        {
          v64 = sub_67B9F0();
          *(_DWORD *)v64 = 2;
          *(_DWORD *)(v64 + 176) = 453;
          v65 = *(_QWORD *)(v3 + 96);
          *(_QWORD *)(v64 + 16) = v3;
          *(_QWORD *)(v64 + 96) = v65;
          if ( !*(_QWORD *)(v3 + 40) )
            *(_QWORD *)(v3 + 40) = v64;
          v66 = *(_QWORD *)(v3 + 48);
          if ( v66 )
            *(_QWORD *)(v66 + 8) = v64;
          *(_QWORD *)(v3 + 48) = v64;
        }
        if ( dword_4F04C64 > 0 )
        {
          v77 = v14;
          v34 = 0;
          v87 = v14 == 1;
          v35 = dword_4F04C64;
          v36 = 0;
          v79 = v14 - v82;
          v37 = 776LL * dword_4F04C64;
          do
          {
            if ( (unsigned int)sub_67B7E0(v37 + qword_4F04C68[0], s, &v88, &qword_4CFDE90, v87) )
            {
              if ( ++v36 > v82 && v80 && v36 <= v79 )
              {
                ++v34;
              }
              else
              {
                if ( v34 )
                {
                  v47 = (_DWORD *)sub_67B9F0();
                  *v47 = 2;
                  v48 = v47;
                  v47[44] = 1150;
                  v49 = *(_QWORD *)(v3 + 96);
                  v48[2] = v3;
                  v48[12] = v49;
                  if ( !*(_QWORD *)(v3 + 40) )
                    *(_QWORD *)(v3 + 40) = v48;
                  v50 = *(_QWORD *)(v3 + 48);
                  if ( v50 )
                    *(_QWORD *)(v50 + 8) = v48;
                  v51 = qword_4D039F0;
                  *(_QWORD *)(v3 + 48) = v48;
                  if ( !v51 || dword_4D03A00 == -1 )
                  {
                    v84 = v48;
                    v51 = sub_823020((unsigned int)dword_4D03A00, 40);
                    v48 = v84;
                  }
                  else
                  {
                    qword_4D039F0 = *(_QWORD *)(v51 + 8);
                  }
                  *(_QWORD *)(v51 + 8) = 0;
                  *(_DWORD *)v51 = 0;
                  *(_QWORD *)(v51 + 16) = v34;
                  if ( !v48[23] )
                    v48[23] = v51;
                  v52 = v48[24];
                  if ( v52 )
                    *(_QWORD *)(v52 + 8) = v51;
                  v48[24] = v51;
                }
                v81 = v88;
                v83 = *(_QWORD *)s;
                v42 = (_DWORD *)sub_67B9F0();
                *v42 = 2;
                v43 = v42;
                v42[44] = v81;
                v44 = *(_QWORD *)(v3 + 96);
                v43[2] = v3;
                v43[12] = v44;
                if ( !*(_QWORD *)(v3 + 40) )
                  *(_QWORD *)(v3 + 40) = v43;
                v45 = *(_QWORD *)(v3 + 48);
                if ( v45 )
                  *(_QWORD *)(v45 + 8) = v43;
                *(_QWORD *)(v3 + 48) = v43;
                v46 = sub_67BB20(4);
                *(_DWORD *)(v46 + 24) = v35;
                *(_QWORD *)(v46 + 16) = v83;
                if ( !v43[23] )
                  v43[23] = v46;
                v38 = v43[24];
                if ( v38 )
                  *(_QWORD *)(v38 + 8) = v46;
                v43[24] = v46;
                v39 = qword_4D039F0;
                if ( !qword_4D039F0 || dword_4D03A00 == -1 )
                  v39 = sub_823020((unsigned int)dword_4D03A00, 40);
                else
                  qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
                *(_DWORD *)v39 = 2;
                v40 = qword_4CFDE90;
                *(_QWORD *)(v39 + 8) = 0;
                *(_QWORD *)(v39 + 16) = v40;
                if ( !v43[23] )
                  v43[23] = v39;
                v41 = v43[24];
                if ( v41 )
                  *(_QWORD *)(v41 + 8) = v39;
                v43[24] = v39;
                v34 = 0;
              }
            }
            v37 -= 776;
            --v35;
          }
          while ( v35 );
          v14 = v77;
        }
        if ( v78 )
        {
          v53 = sub_729AB0(*(unsigned int *)(v3 + 96));
          v54 = sub_723640(v53, 0, 0);
          v55 = sub_724840((unsigned int)dword_4D03A00, v54);
          v56 = (_DWORD *)sub_67B9F0();
          *v56 = 2;
          v57 = v56;
          v56[44] = (v14 != 1) + 1063;
          v58 = *(_QWORD *)(v3 + 96);
          v57[2] = v3;
          v57[12] = v58;
          if ( !*(_QWORD *)(v3 + 40) )
            *(_QWORD *)(v3 + 40) = v57;
          v59 = *(_QWORD *)(v3 + 48);
          if ( v59 )
            *(_QWORD *)(v59 + 8) = v57;
          *(_QWORD *)(v3 + 48) = v57;
          if ( v55 )
          {
            v60 = qword_4D039F0;
            if ( !qword_4D039F0 || dword_4D03A00 == -1 )
              v60 = sub_823020((unsigned int)dword_4D03A00, 40);
            else
              qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
            *(_QWORD *)(v60 + 8) = 0;
            *(_DWORD *)v60 = 3;
            *(_QWORD *)(v60 + 16) = v55;
            if ( !v57[23] )
              v57[23] = v60;
            v61 = v57[24];
            if ( v61 )
              *(_QWORD *)(v61 + 8) = v60;
            v57[24] = v60;
          }
        }
        goto LABEL_44;
      }
LABEL_43:
      if ( !v14 )
        goto LABEL_44;
      v78 = 0;
      goto LABEL_98;
    }
    goto LABEL_90;
  }
LABEL_50:
  a1 = *(unsigned __int8 *)(v3 + 180);
  if ( (unsigned __int8)a1 <= 7u )
  {
    v16 = (unsigned __int8)a1;
    v17 = unk_4F07480 <= (unsigned __int8)a1;
    a1 = 7;
    if ( !v17 )
      a1 = v16;
  }
  a2 = (FILE *)&qword_4F074E8;
  sub_67B780(a1, &qword_4F074E8);
  v2 = 0;
  result = *(unsigned __int8 *)(v3 + 180);
LABEL_2:
  v6 = (unsigned int)(unsigned __int8)result - 9;
  if ( (unsigned int)v6 <= 2 )
  {
    sub_7AFBD0(a1, a2, v6, v2);
    sub_7235F0(*(unsigned __int8 *)(v3 + 180));
  }
  if ( unk_4F074B0 + unk_4F074B8 >= unk_4F07478 )
  {
    v71 = sub_67C860(1508);
    v72 = (__int64)"%s\n";
    v73 = (const char *)qword_4F07510;
    fprintf(qword_4F07510, "%s\n", v71);
    v75 = qword_4D04908;
    if ( qword_4D04908 )
    {
      v72 = 1;
      v73 = "C \"\" 0 0 error limit reached\n";
      fwrite("C \"\" 0 0 error limit reached\n", 1u, 0x1Du, qword_4D04908);
    }
    sub_7AFBD0(v73, v72, v74, v75);
    sub_7235F0(9);
  }
  if ( ((unsigned __int8)v2 & (dword_4D03A14 == 0)) != 0 && (_BYTE)result == 5 && unk_4D04728 )
  {
    unk_4D04728 = 0;
    dword_4D03A14 = 1;
    v62 = byte_4F07481[0];
    byte_4F07481[0] = 4;
    v63 = sub_67D610(0xE7Du, &dword_4F077C8, 4u);
    sub_6837D0(v63);
    result = (__int64)byte_4F07481;
    unk_4D04728 = 1;
    byte_4F07481[0] = v62;
  }
  if ( dword_4D03A00 != -1 )
    return sub_67C610((_QWORD *)v3);
  return result;
}
