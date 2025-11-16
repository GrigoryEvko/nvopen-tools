// Function: sub_7C8410
// Address: 0x7c8410
//
__int64 __fastcall sub_7C8410(unsigned int a1, int a2, int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r15d
  bool v10; // zf
  bool v11; // r13
  bool v12; // al
  bool v13; // r9
  bool v14; // dl
  int v15; // r11d
  _QWORD *v16; // rdi
  int v17; // r11d
  char v18; // al
  int v19; // eax
  unsigned int v20; // r8d
  bool v21; // cl
  unsigned int v22; // eax
  char v23; // si
  int v24; // eax
  __int64 v25; // rax
  char i; // dl
  int v27; // eax
  __int64 v28; // rax
  unsigned int v29; // eax
  int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned int v38; // edi
  __int64 v39; // rcx
  __int64 v40; // rax
  int v41; // eax
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 *v44; // rdi
  __m128i *v45; // rsi
  int v46; // r11d
  unsigned int v47; // r8d
  __int64 v48; // rdx
  int v49; // eax
  int v50; // eax
  __int64 v51; // rax
  int v52; // eax
  __int64 v53; // rax
  __int64 v54; // rax
  int v55; // eax
  __int64 v56; // rcx
  __int64 v57; // rax
  unsigned int v58; // [rsp+10h] [rbp-80h]
  unsigned int v59; // [rsp+18h] [rbp-78h]
  bool v60; // [rsp+18h] [rbp-78h]
  bool v61; // [rsp+18h] [rbp-78h]
  __int64 v62; // [rsp+18h] [rbp-78h]
  __int64 v63; // [rsp+18h] [rbp-78h]
  bool v64; // [rsp+20h] [rbp-70h]
  bool v65; // [rsp+20h] [rbp-70h]
  bool v66; // [rsp+20h] [rbp-70h]
  bool v67; // [rsp+20h] [rbp-70h]
  unsigned int v68; // [rsp+20h] [rbp-70h]
  unsigned int v69; // [rsp+20h] [rbp-70h]
  bool v70; // [rsp+28h] [rbp-68h]
  int v71; // [rsp+28h] [rbp-68h]
  bool v72; // [rsp+28h] [rbp-68h]
  unsigned int v73; // [rsp+28h] [rbp-68h]
  bool v74; // [rsp+28h] [rbp-68h]
  unsigned int v75; // [rsp+28h] [rbp-68h]
  bool v76; // [rsp+28h] [rbp-68h]
  int v77; // [rsp+30h] [rbp-60h]
  int v78; // [rsp+30h] [rbp-60h]
  int v79; // [rsp+30h] [rbp-60h]
  int v80; // [rsp+30h] [rbp-60h]
  int v81; // [rsp+30h] [rbp-60h]
  __int64 v82; // [rsp+38h] [rbp-58h]
  __int64 v83; // [rsp+40h] [rbp-50h]
  int v85; // [rsp+48h] [rbp-48h]
  int v86; // [rsp+48h] [rbp-48h]
  const char *v87; // [rsp+48h] [rbp-48h]
  int v88; // [rsp+48h] [rbp-48h]
  int v89; // [rsp+54h] [rbp-3Ch] BYREF
  _QWORD v90[7]; // [rsp+58h] [rbp-38h] BYREF

  v6 = 0;
  *a3 = 0;
  if ( dword_4F077C4 == 2 )
  {
    if ( word_4F06418[0] != 1 )
      goto LABEL_4;
    if ( qword_4D04A18 )
    {
      v6 = word_4D04A10 & 1;
      v10 = (word_4D04A10 & 0x2000) == 0;
      *a3 = (word_4D04A10 & 0x2000) != 0;
      if ( !v10
        || ((a1 & 0x1C) == 0 || !(unsigned int)sub_7AC1A0(a1, dword_4F07508))
        && ((a1 & 0x2080) == 0 || !(unsigned int)sub_7ADC90(a1)) )
      {
        goto LABEL_8;
      }
      *a3 = 1;
      v11 = 0;
      sub_885B10(&qword_4D04A00);
      v83 = 0;
      v12 = 0;
      goto LABEL_16;
    }
    if ( (word_4D04A10 & 0x200) == 0 )
    {
LABEL_4:
      if ( !(unsigned int)sub_7C0F00(a1 & 0xFFFFFFE3, 0, (__int64)a3, a4, a5, a6) || word_4F06418[0] != 1 )
        goto LABEL_5;
    }
    if ( (word_4D04A10 & 1) == 0 )
    {
LABEL_5:
      if ( (word_4D04A10 & 0x2000) != 0 )
        *a3 = 1;
      v6 = 0;
      goto LABEL_8;
    }
    v83 = 0;
    v90[0] = qword_4D04A08;
    v82 = xmmword_4D04A20.m128i_i64[0];
    if ( (unk_4D04A12 & 2) != 0 )
    {
      v83 = xmmword_4D04A20.m128i_i64[0];
      v82 = 0;
    }
    v11 = (word_4D04A10 & 0x400) != 0;
    v13 = (unk_4D04A12 & 2) != 0;
    v14 = (unk_4D04A12 & 8) != 0;
    v15 = *a3 | ((word_4D04A10 & 0x2000) != 0);
    v6 = v13;
    *a3 = v15;
    if ( v15 )
    {
      v16 = qword_4D04A18;
      v17 = 0;
      goto LABEL_23;
    }
    if ( (a1 & 0x1C) != 0 )
    {
      v60 = v14;
      v66 = v13;
      v30 = sub_7AC1A0(a1, &dword_4F063F8);
      v13 = v66;
      v14 = v60;
      if ( v30 )
      {
        *a3 = 1;
        v17 = 0;
        v16 = qword_4D04A18;
        goto LABEL_23;
      }
    }
    if ( (unk_4D04A12 & 0x20) != 0 )
    {
      if ( v83 )
      {
        if ( !v11 )
        {
          v64 = v14;
          v70 = v13;
          v19 = sub_8D3A70(v83);
          v13 = v70;
          v14 = v64;
          if ( !v19 )
          {
            v41 = sub_8D2870(v83);
            v13 = v70;
            v14 = v64;
            if ( !v41 )
            {
              if ( dword_4F04C44 == -1
                && (v42 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v42 + 6) & 6) == 0)
                && *(_BYTE *)(v42 + 4) != 12
                || (v52 = sub_8DBE70(v83), v13 = v70, v14 = v64, !v52) )
              {
                sub_685360(0xC3Cu, &dword_4F063F8, v83);
                v43 = 16;
                v44 = &qword_4D04A00;
                v45 = xmmword_4F06660;
                v17 = 1;
                while ( v43 )
                {
                  *(_DWORD *)v44 = v45->m128i_i32[0];
                  v45 = (__m128i *)((char *)v45 + 4);
                  v44 = (__int64 *)((char *)v44 + 4);
                  --v43;
                }
                HIBYTE(word_4D04A10) |= 0x20u;
                qword_4D04A08 = *(_QWORD *)dword_4F07508;
                *a3 = 1;
                v16 = qword_4D04A18;
                goto LABEL_23;
              }
            }
          }
        }
      }
    }
    v20 = dword_3C19800[a2];
    v21 = (word_4D04A10 & 0x800) != 0;
    v22 = v20;
    if ( (a1 & 0x100000) != 0 )
    {
      BYTE1(v22) = BYTE1(v20) | 0x40;
      v20 = v22;
    }
    if ( (a1 & 0x4000000) != 0 )
      v20 |= 0x40000u;
    if ( v83 && (v20 & 0x8000000) != 0 && *(_BYTE *)(v83 + 140) == 14 )
    {
      v58 = v20;
      v61 = (word_4D04A10 & 0x800) != 0;
      v67 = v14;
      v72 = v13;
      v40 = sub_7D0530(v83);
      v20 = v58;
      v21 = v61;
      v14 = v67;
      v83 = v40;
      v13 = v72;
    }
    v23 = word_4D04A10;
    if ( (word_4D04A10 & 4) != 0 )
    {
      if ( v11 )
      {
LABEL_52:
        v17 = v83 != 0;
        goto LABEL_53;
      }
      v34 = sub_7D4600(unk_4F07288, &qword_4D04A00, v20);
      if ( a2 != 2 && !v34 )
      {
        v46 = a1 & 0x80000;
        if ( (a1 & 0x80000) == 0 )
        {
          if ( a2 == 1 )
          {
            v47 = 470;
          }
          else if ( a2 == 7 || a2 == 4 )
          {
            v47 = 1019;
          }
          else
          {
            v47 = 282;
          }
          v48 = *(_QWORD *)(qword_4D04A00 + 8);
          if ( (a1 & 0x8000000) == 0
            || (v73 = v47,
                v87 = *(const char **)(qword_4D04A00 + 8),
                v49 = strcmp(v87, "gets"),
                v48 = (__int64)v87,
                v46 = a1 & 0x80000,
                v47 = v73,
                v49) )
          {
            v88 = v46;
            sub_6851A0(v47, v90, v48);
            v16 = qword_4D04A18;
            v17 = v88;
          }
          else
          {
            v16 = qword_4D04A18;
            v17 = 0;
          }
          goto LABEL_23;
        }
      }
      goto LABEL_94;
    }
    v17 = v14;
    if ( !v14 )
    {
      if ( !v6 )
      {
        if ( !v82 )
          goto LABEL_53;
        if ( v11 )
          goto LABEL_52;
        goto LABEL_97;
      }
      v25 = v83;
      for ( i = *(_BYTE *)(v83 + 140); i == 12; i = *(_BYTE *)(v25 + 140) )
        v25 = *(_QWORD *)(v25 + 160);
      if ( !i )
        goto LABEL_53;
      if ( !v21 )
      {
        v59 = v20;
        v65 = v13;
        v71 = v17;
        v27 = sub_8D23B0(v83);
        v17 = v71;
        v13 = v65;
        v20 = v59;
        if ( v27 )
        {
          if ( (unsigned __int8)(*(_BYTE *)(v83 + 140) - 9) <= 2u )
          {
            v28 = *(_QWORD *)(*(_QWORD *)(v83 + 168) + 152LL);
            if ( !v28 || (*(_BYTE *)(v28 + 29) & 0x20) != 0 )
            {
              if ( (a1 & 0x80000) == 0 )
              {
                v29 = sub_67F240();
                sub_685A50(v29, &dword_4F063F8, (FILE *)v83, 8u);
                v16 = qword_4D04A18;
                v17 = 0;
                goto LABEL_23;
              }
LABEL_53:
              v16 = qword_4D04A18;
              goto LABEL_23;
            }
          }
        }
        v23 = word_4D04A10;
      }
    }
    v31 = v23 & 0x20;
    if ( (_DWORD)v31 && a2 != 12 && v13 )
    {
      v32 = 0;
      if ( (unsigned __int8)(*(_BYTE *)(v83 + 140) - 9) <= 2u )
        v32 = *(_QWORD *)(*(_QWORD *)v83 + 96LL);
    }
    else
    {
      v32 = 0;
    }
    if ( v11 )
      goto LABEL_52;
    if ( a2 == 9 && v13 )
    {
      if ( dword_4F077BC )
      {
        if ( !(_DWORD)qword_4F077B4 )
        {
          if ( qword_4F077A8 )
          {
            v33 = qword_4F04C68[0] + 776LL * dword_4F04C64;
            if ( *(_BYTE *)(v33 + 4) == 6 && *(char *)(*(_QWORD *)(v33 + 208) + 177LL) < 0 )
            {
              v63 = v32;
              v69 = v20;
              v76 = v13;
              v81 = v17;
              v55 = sub_8DBE70(v83);
              v17 = v81;
              v13 = v76;
              v20 = v69;
              v32 = v63;
              if ( !v55 )
              {
                v57 = sub_7D2400(v83, v31, v63, v56, v69, v76);
                v17 = v81;
                v13 = v76;
                v20 = v69;
                v32 = v63;
                v83 = v57;
              }
            }
          }
        }
      }
      if ( !dword_4D044A0 )
      {
        if ( v32 )
        {
          v16 = *(_QWORD **)(v32 + 24);
          if ( v16 )
            goto LABEL_90;
        }
        goto LABEL_139;
      }
    }
    else
    {
      if ( !dword_4D044A0 )
      {
        if ( !v32 )
        {
LABEL_138:
          if ( v13 )
          {
LABEL_139:
            v79 = v17;
            v51 = sub_7D2AC0(&qword_4D04A00, v83, v20);
            v17 = v79;
            if ( v51 )
              goto LABEL_94;
            goto LABEL_102;
          }
          goto LABEL_97;
        }
        goto LABEL_137;
      }
      if ( !v6 )
      {
        if ( !v32 )
          goto LABEL_97;
        goto LABEL_125;
      }
    }
    v62 = v32;
    v68 = v20;
    v74 = v13;
    v78 = v17;
    v50 = sub_8D2870(v83);
    v17 = v78;
    v13 = v74;
    v20 = v68;
    v32 = v62;
    if ( !v50 )
    {
      if ( !v62 )
        goto LABEL_139;
LABEL_137:
      v16 = *(_QWORD **)(v32 + 24);
      if ( !v16 )
        goto LABEL_138;
LABEL_90:
      qword_4D04A18 = v16;
      v17 = 1;
      v89 = 0;
LABEL_24:
      v18 = *((_BYTE *)v16 + 80);
      if ( v18 == 16 )
      {
        v16 = *(_QWORD **)v16[11];
        v18 = *((_BYTE *)v16 + 80);
      }
      if ( v18 == 24 )
      {
        v16 = (_QWORD *)v16[11];
        v18 = *((_BYTE *)v16 + 80);
      }
      if ( v18 == 19 )
      {
        v86 = v17;
        sub_7BF840((__int64)v16, a1, &v89);
        v17 = v86;
        if ( v89 )
        {
          *a3 = 1;
          if ( (a1 & 0x2080) == 0 || !(unsigned int)sub_7ADC90(a1) )
            goto LABEL_31;
          goto LABEL_55;
        }
      }
LABEL_29:
      if ( (a1 & 0x2080) == 0 || (v85 = v17, v24 = sub_7ADC90(a1), v17 = v85, !v24) )
      {
        if ( v17 )
        {
          v6 = v17;
          goto LABEL_8;
        }
        goto LABEL_31;
      }
LABEL_55:
      *a3 = 1;
LABEL_31:
      sub_885B10(&qword_4D04A00);
      if ( !v6 )
      {
        unk_4D04A12 &= ~2u;
        v6 = 1;
        xmmword_4D04A20.m128i_i64[0] = v82;
        goto LABEL_33;
      }
      v12 = v83 != 0;
LABEL_16:
      xmmword_4D04A20.m128i_i64[0] = v83;
      unk_4D04A12 = (2 * v12) | unk_4D04A12 & 0xFD;
LABEL_33:
      HIBYTE(word_4D04A10) = (4 * v11) | HIBYTE(word_4D04A10) & 0xFB;
      *a3 = 1;
LABEL_8:
      *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
      return v6;
    }
    v54 = sub_7D36A0(&qword_4D04A00, v83);
    v17 = v78;
    v13 = v74;
    v20 = v68;
    v32 = v62;
    if ( v54 )
      goto LABEL_94;
    if ( !v62 )
      goto LABEL_102;
LABEL_125:
    v16 = *(_QWORD **)(v32 + 24);
    if ( v16 )
      goto LABEL_90;
LABEL_97:
    if ( v82 && !v13 )
    {
      v35 = v82;
      if ( (*(_BYTE *)(v82 + 124) & 1) != 0 )
      {
        v75 = v20;
        v80 = v17;
        v53 = sub_735B70(v82);
        v20 = v75;
        v17 = v80;
        v35 = v53;
      }
      v77 = v17;
      v36 = sub_7D4A40(&qword_4D04A00, v35, v20);
      v17 = v77;
      if ( v36 )
        goto LABEL_94;
    }
LABEL_102:
    if ( a2 != 2 && (a1 & 0x80000) == 0 && (word_4D04A10 & 0x2000) == 0 )
    {
      v37 = *(_QWORD *)(qword_4D04A00 + 8);
      if ( v17 )
      {
        sub_6851A0(0x47Au, v90, v37);
        v16 = qword_4D04A18;
        v17 = 0;
      }
      else
      {
        if ( a2 == 1 || a2 == 14 )
        {
          v38 = 471;
        }
        else if ( a2 == 7 || a2 == 4 )
        {
          v38 = 1018;
        }
        else
        {
          v38 = (dword_4F077C4 != 2) + 135;
        }
        if ( v6 )
          v39 = *(_QWORD *)v83;
        else
          v39 = *(_QWORD *)v82;
        sub_686A10(v38, v90, v37, v39);
        v16 = qword_4D04A18;
        v17 = 0;
      }
LABEL_23:
      v89 = 0;
      if ( !v16 )
        goto LABEL_29;
      goto LABEL_24;
    }
LABEL_94:
    v16 = qword_4D04A18;
    v17 = 1;
    goto LABEL_23;
  }
  return v6;
}
