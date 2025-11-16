// Function: sub_7D2AC0
// Address: 0x7d2ac0
//
__int64 __fastcall sub_7D2AC0(_QWORD *a1, const char *a2, unsigned int a3)
{
  const char *v3; // r13
  int v4; // r12d
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  char v12; // al
  char v13; // al
  __int64 v14; // r13
  int v15; // eax
  __int64 v16; // r15
  _QWORD *v18; // r14
  __int64 v19; // rax
  const char *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // r12
  __int64 v27; // rax
  char v28; // dl
  char v29; // al
  __int64 v30; // rbx
  char v31; // dl
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rdi
  char v36; // dl
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  _QWORD *v40; // rax
  char v41; // al
  char v42; // al
  int v43; // eax
  __int64 v44; // rax
  _QWORD *v45; // [rsp+18h] [rbp-A8h]
  char v46; // [rsp+20h] [rbp-A0h]
  __int64 v47; // [rsp+20h] [rbp-A0h]
  int v48; // [rsp+28h] [rbp-98h]
  int v49; // [rsp+2Ch] [rbp-94h]
  _QWORD *v50; // [rsp+38h] [rbp-88h] BYREF
  _DWORD v51[2]; // [rsp+40h] [rbp-80h] BYREF
  int v52; // [rsp+48h] [rbp-78h]
  int v53; // [rsp+4Ch] [rbp-74h]
  int v54; // [rsp+50h] [rbp-70h]
  int v55; // [rsp+54h] [rbp-6Ch]
  int v56; // [rsp+58h] [rbp-68h]
  int v57; // [rsp+5Ch] [rbp-64h]
  int v58; // [rsp+60h] [rbp-60h]
  int v59; // [rsp+64h] [rbp-5Ch]
  int v60; // [rsp+68h] [rbp-58h]
  int v61; // [rsp+6Ch] [rbp-54h]
  int v62; // [rsp+70h] [rbp-50h]
  int v63; // [rsp+74h] [rbp-4Ch]
  int v64; // [rsp+78h] [rbp-48h]
  int v65; // [rsp+7Ch] [rbp-44h]
  int v66; // [rsp+80h] [rbp-40h]
  int v67; // [rsp+84h] [rbp-3Ch]
  int v68; // [rsp+88h] [rbp-38h]

  v3 = a2;
  v4 = a3;
  v5 = (__int64)a1;
  v51[1] = a3 & 1;
  v51[0] = 0;
  v52 = (a3 >> 1) & 1;
  v53 = (a3 >> 11) & 1;
  v54 = (a3 >> 27) & 1;
  v55 = (a3 >> 4) & 1;
  v56 = (a3 >> 26) & 1;
  v57 = (a3 >> 17) & 1;
  v58 = (a3 >> 20) & 1;
  v59 = (a3 >> 14) & 1;
  v60 = HIBYTE(a3) & 1;
  v61 = HIWORD(a3) & 1;
  v62 = (a3 >> 12) & 1;
  v63 = (a3 >> 23) & 1;
  v64 = (a3 >> 22) & 1;
  v65 = (a3 >> 10) & 1;
  v66 = (a3 >> 2) & 1;
  v67 = (a3 >> 15) & 1;
  v68 = (a3 >> 13) & 1;
  if ( (a3 & 3) == 0 && (a3 & 0x10000) != 0 )
  {
    if ( dword_4D0489C )
    {
      v18 = *(_QWORD **)a2;
      if ( *(_QWORD *)a2 )
      {
        v19 = a1[6];
        if ( v19 )
        {
          a2 = *(const char **)(v19 + 24);
          if ( a2 )
          {
            v20 = *(const char **)(*a1 + 8LL);
            if ( !strcmp(v20, a2) )
            {
              if ( (unsigned __int8)(v3[140] - 9) > 2u )
                goto LABEL_132;
              goto LABEL_35;
            }
            if ( dword_4F077BC )
            {
              if ( !(_DWORD)qword_4F077B4 && (unsigned __int8)(v3[140] - 9) <= 2u )
              {
                a2 = *(const char **)(*v18 + 8LL);
                if ( !strcmp(v20, a2) )
                {
LABEL_35:
                  v47 = *(_QWORD *)(v18[12] + 8LL);
                  if ( v47 )
                  {
LABEL_36:
                    v21 = v47;
LABEL_37:
                    v50 = (_QWORD *)v21;
                    goto LABEL_38;
                  }
LABEL_132:
                  *((_BYTE *)a1 + 17) |= 0x10u;
                  v47 = (__int64)v18;
                  goto LABEL_36;
                }
              }
            }
          }
        }
      }
    }
    v50 = 0;
  }
  v6 = sub_8D2220(v3);
  v11 = v6;
  if ( !v54 )
    goto LABEL_11;
  v12 = *(_BYTE *)(v6 + 140);
  if ( v12 == 14 )
  {
    v11 = sub_7D0530(v11);
LABEL_11:
    v12 = *(_BYTE *)(v11 + 140);
    if ( v12 == 14 )
      goto LABEL_12;
  }
  if ( v12 != 12 )
  {
    v13 = *(_BYTE *)(v11 + 177);
    v49 = 0;
    if ( (v13 & 0x20) != 0 )
    {
      v49 = 1;
      if ( !*(_QWORD *)(*(_QWORD *)(v11 + 168) + 256LL) && (v13 & 0xD0) != 0x10 )
      {
        v51[0] = 1;
        v49 = 0;
      }
    }
    goto LABEL_13;
  }
  v11 = (__int64)v3;
LABEL_12:
  v49 = 1;
  v11 = sub_7CFE40(v11, (__int64)a2, v7, v8, v9, v10);
LABEL_13:
  v14 = *(_QWORD *)(*(_QWORD *)v11 + 96LL);
  v45 = *(_QWORD **)v11;
  v46 = *(_BYTE *)(v14 + 180) & 1;
  v15 = 0;
  if ( (a1[2] & 0x10) != 0 )
  {
    v15 = v64;
    if ( v64 )
      v15 = sub_8DBE70(a1[7]);
  }
  v16 = a1[3];
  v50 = (_QWORD *)v16;
  if ( (a1[2] & 0x2040) != 0 )
    return 0;
  if ( v16 )
    goto LABEL_18;
  if ( v15 )
    goto LABEL_55;
  if ( (a1[2] & 8) != 0 && *((_BYTE *)a1 + 56) == 15 )
  {
    v23 = *(_QWORD *)(v14 + 32);
    v50 = (_QWORD *)v23;
    if ( v23 )
    {
      if ( sub_7CF4E0(v51, v11, v23, v23) )
      {
        v47 = (__int64)v50;
        if ( v50 )
          goto LABEL_38;
      }
      else
      {
        v50 = 0;
      }
    }
    goto LABEL_55;
  }
  v50 = (_QWORD *)sub_883800(v14 + 192, *a1);
  v24 = (__int64)v50;
  if ( !v50 )
    goto LABEL_55;
  v25 = 0;
  v48 = v4;
  v26 = 0;
  do
  {
    v29 = *(_BYTE *)(v24 + 80);
    v30 = v24;
    if ( v29 == 16 )
    {
      v30 = **(_QWORD **)(v24 + 88);
      v29 = *(_BYTE *)(v30 + 80);
    }
    if ( v29 == 24 )
      v30 = *(_QWORD *)(v30 + 88);
    if ( !sub_7CF4E0(v51, v11, v24, v30) || v49 && !sub_7CEAF0((__int64)v50, v48, (__int64)a1) )
    {
LABEL_76:
      v27 = (__int64)v50;
      goto LABEL_70;
    }
    if ( v46 )
    {
      v27 = (__int64)v50;
      if ( *((_BYTE *)v50 + 80) != 16 )
        goto LABEL_65;
      v31 = *((_BYTE *)v50 + 96);
      if ( (v31 & 0x10) == 0 )
      {
        if ( !v62 )
        {
          if ( v52 )
          {
LABEL_126:
            v47 = v27;
            v5 = (__int64)a1;
            goto LABEL_38;
          }
LABEL_66:
          v28 = *(_BYTE *)(v30 + 80);
          if ( (unsigned __int8)(v28 - 4) <= 2u || v28 == 3 && *(_BYTE *)(v30 + 104) )
          {
            if ( v25 )
            {
              if ( *(_BYTE *)(v25 + 80) == 24 )
                v25 = v27;
            }
            else
            {
              v25 = v27;
            }
          }
          else
          {
            if ( *(_BYTE *)(v27 + 80) != 3 )
              goto LABEL_126;
            v26 = v27;
          }
          goto LABEL_70;
        }
        goto LABEL_81;
      }
      if ( !sub_7CEAF0(v30, v48, (__int64)a1) )
        goto LABEL_76;
    }
    v27 = (__int64)v50;
    if ( !v62 || *((_BYTE *)v50 + 80) != 16 )
      goto LABEL_65;
    v31 = *((_BYTE *)v50 + 96);
LABEL_81:
    if ( (v31 & 4) == 0 )
      goto LABEL_70;
LABEL_65:
    if ( !v52 )
      goto LABEL_66;
    if ( *(_BYTE *)(v27 + 80) != 3 )
      goto LABEL_126;
    if ( v26 && *(_BYTE *)(v26 + 80) == 24 )
      v26 = 0;
    if ( !v25 )
      v25 = v27;
    if ( !v26 )
      v26 = v27;
LABEL_70:
    v24 = *(_QWORD *)(v27 + 32);
    v50 = (_QWORD *)v24;
  }
  while ( v24 );
  v21 = v26;
  v47 = v26;
  v39 = v25;
  v16 = 0;
  v5 = (__int64)a1;
  v4 = v48;
  if ( v21 )
    goto LABEL_37;
  if ( v39 )
  {
    v50 = (_QWORD *)v39;
    v47 = v39;
    goto LABEL_38;
  }
LABEL_55:
  if ( v49 && !v55 )
  {
    v47 = sub_7D0430((_QWORD *)v11, v4, v5);
    sub_886160(v47);
    v50 = (_QWORD *)v47;
    goto LABEL_58;
  }
  v47 = (__int64)v50;
  if ( dword_4F077C4 != 2 )
  {
LABEL_58:
    if ( v47 )
      goto LABEL_38;
    goto LABEL_59;
  }
  v32 = *(_QWORD *)v5;
  if ( *(_QWORD *)v5 == *v45 )
  {
    v47 = *(_QWORD *)(v14 + 8);
    v50 = (_QWORD *)v47;
    if ( v47
      || (*(_BYTE *)(v11 + 141) & 0x20) == 0
      && (*(_BYTE *)(v11 + 177) & 0x20) == 0
      && (v47 = sub_5F1EF0((__int64)v45), (v50 = (_QWORD *)v47) != 0) )
    {
      *(_QWORD *)v5 = *(_QWORD *)v47;
      goto LABEL_38;
    }
    if ( (*(_BYTE *)(v5 + 16) & 0x20) == 0 )
      goto LABEL_147;
    goto LABEL_139;
  }
  if ( (*(_BYTE *)(v5 + 16) & 0x20) != 0 )
  {
LABEL_139:
    v40 = *(_QWORD **)(v14 + 24);
    v50 = v40;
    if ( v40
      || (*(_BYTE *)(v11 + 141) & 0x20) == 0
      && (*(_BYTE *)(v11 + 177) & 0x20) == 0
      && (v40 = (_QWORD *)sub_5F2480((__int64)v45, (_QWORD *)v5), (v50 = v40) != 0) )
    {
      if ( *(_QWORD *)v5 == *v40 )
      {
        v47 = (__int64)v50;
        goto LABEL_38;
      }
      goto LABEL_59;
    }
LABEL_147:
    if ( v62 )
      goto LABEL_152;
    v41 = *(_BYTE *)(v5 + 16);
    if ( (v41 & 0x10) == 0 || v61 )
    {
      v42 = v41 & 0x20;
      goto LABEL_151;
    }
    v47 = (__int64)sub_7D2920(v5, v11);
    v50 = (_QWORD *)v47;
    if ( v47 )
      goto LABEL_38;
    if ( !v62 )
    {
      v42 = *(_BYTE *)(v5 + 16) & 0x20;
LABEL_151:
      if ( !v42 )
      {
        v32 = *(_QWORD *)v5;
        goto LABEL_104;
      }
LABEL_152:
      v47 = (__int64)v50;
      goto LABEL_58;
    }
LABEL_59:
    *(_QWORD *)(v5 + 24) = 0;
    return v16;
  }
  if ( !v50 )
    goto LABEL_147;
  if ( v62 )
    goto LABEL_38;
LABEL_104:
  v33 = *(_QWORD *)(v32 + 24);
  v34 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( qword_4F04C68[0] == v34 )
  {
LABEL_153:
    v35 = 0;
    v43 = 0;
    goto LABEL_154;
  }
  v35 = 0;
  while ( 2 )
  {
    v36 = *(_BYTE *)(v34 + 4);
    if ( v36 != 6 )
    {
      if ( v36 != 7 )
      {
LABEL_114:
        if ( v33 )
        {
          while ( *(_DWORD *)(v33 + 40) == *(_DWORD *)v34 )
          {
            v35 = v33;
            if ( !*(_QWORD *)(v33 + 8) )
            {
              v33 = 0;
              break;
            }
            v33 = *(_QWORD *)(v33 + 8);
          }
        }
      }
      v34 -= 776;
      if ( qword_4F04C68[0] == v34 )
        goto LABEL_153;
      continue;
    }
    break;
  }
  v37 = *(_QWORD *)(v34 + 208);
  if ( v11 != v37 )
  {
    if ( !v37 )
      goto LABEL_114;
    if ( !dword_4F07588 )
      goto LABEL_114;
    v38 = *(_QWORD *)(v37 + 32);
    if ( *(_QWORD *)(v11 + 32) != v38 || !v38 )
      goto LABEL_114;
  }
  v43 = 1;
LABEL_154:
  if ( v55 )
    goto LABEL_158;
  if ( v54 )
  {
    if ( v59 )
      goto LABEL_157;
LABEL_158:
    if ( v57 || v58 )
      sub_886B00(v11, v5, v4, 1, 1, 0, 0, 1, v43, v35, (__int64)&v50, 0);
    else
      sub_886B00(v11, v5, v4, 1, 1, 0, 0, *(_BYTE *)(v14 + 180) & 1, v43, v35, (__int64)&v50, 0);
    v47 = (__int64)v50;
    goto LABEL_58;
  }
  if ( (*(_BYTE *)(v14 + 180) & 1) == 0 )
  {
LABEL_157:
    if ( !dword_4F077BC )
      goto LABEL_158;
  }
  if ( !v51[0] )
    goto LABEL_158;
  if ( v57 || v58 )
    sub_886B00(v11, v5, v4, 1, 1, 0, 0, 1, v43, v35, (__int64)&v50, 0);
  else
    sub_886B00(v11, v5, v4, 1, 1, 0, 0, *(_BYTE *)(v14 + 180) & 1, v43, v35, (__int64)&v50, 0);
  v44 = (__int64)v50;
  v47 = (__int64)v50;
  if ( !v50 )
  {
LABEL_178:
    v47 = sub_7D0430((_QWORD *)v11, v4 | 0x800000u, v5);
    v50 = (_QWORD *)v47;
    goto LABEL_58;
  }
  if ( *((_BYTE *)v50 + 80) == 16 )
    v44 = *(_QWORD *)v50[11];
  if ( *(_BYTE *)(v44 + 80) == 24 )
    v44 = *(_QWORD *)(v44 + 88);
  if ( (*(_BYTE *)(v44 + 84) & 2) != 0 )
  {
    v50 = 0;
    goto LABEL_178;
  }
LABEL_38:
  if ( v56 || !*(_QWORD *)(v47 + 72) || *(char *)(v47 + 81) < 0 )
  {
    v16 = (__int64)v50;
    goto LABEL_121;
  }
  v16 = sub_7D2A80(v47);
  if ( v50 == (_QWORD *)v16 )
    goto LABEL_121;
  v22 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 512);
  if ( v22 )
  {
    while ( v50 != (_QWORD *)v22 )
    {
      if ( (*(_BYTE *)(v22 + 81) & 0x10) != 0 )
      {
        v22 = **(_QWORD **)(v22 + 64);
        if ( v22 )
          continue;
      }
      goto LABEL_133;
    }
    *(_QWORD *)(v5 + 24) = v22;
    v16 = v22;
LABEL_18:
    if ( *(_BYTE *)(v16 + 80) == 16 )
      v16 = **(_QWORD **)(v16 + 88);
    else
      v16 = (__int64)v50;
    if ( *(_BYTE *)(v16 + 80) == 24 )
      return *(_QWORD *)(v16 + 88);
    return v16;
  }
LABEL_133:
  v50 = (_QWORD *)v16;
LABEL_121:
  *(_QWORD *)(v5 + 24) = v16;
  if ( v16 )
    goto LABEL_18;
  return v16;
}
