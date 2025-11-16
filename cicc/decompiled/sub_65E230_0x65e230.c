// Function: sub_65E230
// Address: 0x65e230
//
__int64 __fastcall sub_65E230(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r10
  __int64 v9; // r13
  char v10; // al
  _QWORD *v11; // r11
  __int64 v12; // r14
  char v13; // al
  unsigned int v14; // r10d
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r10
  char v19; // dl
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rax
  char v24; // al
  __int64 v25; // rcx
  char v26; // al
  _QWORD *v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // r9
  unsigned int v30; // edx
  _QWORD *v31; // r11
  _QWORD *v32; // rax
  char v33; // al
  __int64 v34; // r13
  __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  int v40; // r10d
  int v41; // eax
  __int64 v42; // r8
  __int64 v43; // r14
  int v44; // eax
  __int64 v45; // rsi
  int v46; // eax
  char v47; // dl
  __int64 v48; // rax
  char v49; // cl
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rax
  char v54; // al
  unsigned __int8 v55; // dl
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  int v59; // eax
  int v60; // eax
  __int64 v61; // rsi
  __int64 v62; // rdi
  __int64 v63; // rax
  __int64 *v64; // rax
  __int64 v65; // rax
  __int64 v66; // rsi
  unsigned int v67; // eax
  __int64 v68; // rcx
  bool v69; // zf
  __int64 v70; // rdx
  char v71; // dl
  char v72; // r14
  _BOOL4 v73; // r14d
  __int64 v74; // rdx
  _QWORD *v75; // r9
  unsigned int v76; // r10d
  _QWORD *v77; // [rsp+8h] [rbp-78h]
  __int64 v78; // [rsp+8h] [rbp-78h]
  _QWORD *v79; // [rsp+8h] [rbp-78h]
  unsigned int v80; // [rsp+10h] [rbp-70h]
  unsigned int v81; // [rsp+10h] [rbp-70h]
  _QWORD *v82; // [rsp+10h] [rbp-70h]
  _QWORD *v83; // [rsp+10h] [rbp-70h]
  char v84; // [rsp+10h] [rbp-70h]
  char v85; // [rsp+10h] [rbp-70h]
  char v86; // [rsp+10h] [rbp-70h]
  unsigned int v87; // [rsp+10h] [rbp-70h]
  int v88; // [rsp+10h] [rbp-70h]
  _QWORD *v89; // [rsp+10h] [rbp-70h]
  _QWORD *v90; // [rsp+10h] [rbp-70h]
  unsigned int v91; // [rsp+18h] [rbp-68h]
  _QWORD *v92; // [rsp+18h] [rbp-68h]
  __int64 v93; // [rsp+18h] [rbp-68h]
  __int64 v94; // [rsp+18h] [rbp-68h]
  __int64 v95; // [rsp+18h] [rbp-68h]
  __int64 v96; // [rsp+18h] [rbp-68h]
  _QWORD *v97; // [rsp+18h] [rbp-68h]
  _QWORD *v98; // [rsp+18h] [rbp-68h]
  _QWORD *v99; // [rsp+18h] [rbp-68h]
  _QWORD *v100; // [rsp+18h] [rbp-68h]
  _QWORD *v101; // [rsp+18h] [rbp-68h]
  _QWORD *v102; // [rsp+18h] [rbp-68h]
  unsigned int v103; // [rsp+18h] [rbp-68h]
  int v104; // [rsp+18h] [rbp-68h]
  __int64 v106; // [rsp+28h] [rbp-58h]
  _QWORD *v108; // [rsp+38h] [rbp-48h]
  __int64 v109; // [rsp+38h] [rbp-48h]
  __int64 v110; // [rsp+38h] [rbp-48h]
  _QWORD *v111; // [rsp+38h] [rbp-48h]
  __int64 v112[7]; // [rsp+48h] [rbp-38h] BYREF

  v112[0] = *(_QWORD *)(a2 + 288);
  v106 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  *(_BYTE *)(a2 + 130) |= 0x40u;
  if ( *(_QWORD *)(a2 + 304) && (*(_BYTE *)(a2 + 123) & 0x10) == 0 )
  {
    *(_BYTE *)(a2 + 124) &= ~0x40u;
    sub_65C470(a2, a2, a3, a4, a5);
    v112[0] = sub_72C930(a2);
  }
  sub_5CEEC0(v112, *(_QWORD **)(a2 + 200), a2);
  v6 = sub_7CFB70(a1, 64);
  v8 = a1[3];
  v9 = v6;
  if ( v8 && *(_BYTE *)(v8 + 80) == 16 )
  {
    v24 = *(_BYTE *)(v8 + 96) & 4;
    if ( dword_4F077BC && *(_BYTE *)(v106 + 4) == 6 )
    {
      if ( !v24 )
        sub_881DB0(a1[3]);
      if ( (*((_BYTE *)a1 + 17) & 0x40) != 0 )
        goto LABEL_40;
      *((_BYTE *)a1 + 16) &= ~0x80u;
      v14 = 0;
      a1[3] = 0;
    }
    else
    {
      if ( v24 )
        goto LABEL_40;
      v10 = *(_BYTE *)(v9 + 80);
      if ( v10 != 3 )
        goto LABEL_8;
      v25 = *(_QWORD *)(v9 + 88);
      if ( *(_BYTE *)(v25 + 140) != 14 )
        goto LABEL_44;
      v14 = 1;
    }
LABEL_41:
    v12 = v112[0];
    v108 = a1 + 1;
    if ( dword_4F077C4 == 2 && (*((_BYTE *)a1 + 17) & 0x20) == 0 )
    {
      v29 = (_QWORD *)v112[0];
      v30 = dword_4F077BC;
      if ( dword_4F077BC )
      {
        v30 = 0;
        if ( *(_BYTE *)(v112[0] + 140) == 12 )
        {
          do
          {
            if ( (unsigned __int8)(*((_BYTE *)v29 + 184) - 6) > 1u )
              break;
            v29 = (_QWORD *)v29[20];
            v30 = 1;
          }
          while ( *((_BYTE *)v29 + 140) == 12 );
        }
      }
      v31 = (_QWORD *)*v29;
      if ( *v29 )
      {
        v31 = (_QWORD *)v31[9];
        if ( v31 )
        {
          v32 = (_QWORD *)v31[11];
          v31 = v29;
          v29 = v32;
        }
      }
      v33 = *((_BYTE *)v29 + 140);
      if ( ((unsigned __int8)(v33 - 9) <= 2u || v33 == 2 && (*((_BYTE *)v29 + 161) & 8) != 0) && !v29[1] )
      {
        v34 = *v29;
        if ( v30 || *(_DWORD *)(v34 + 40) == *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C) )
        {
          if ( qword_4D0495C )
          {
            v98 = v29;
            sub_886060(*v29, a1);
            v35 = v34;
            v72 = *((_BYTE *)v98 + 88);
            sub_877D80(v98, v34);
            v39 = (__int64)v98;
            v40 = 1;
            v73 = (v72 & 4) != 0;
            v36 = v98[11] & 0xFB | (unsigned int)(4 * v73);
            *((_BYTE *)v98 + 88) = v98[11] & 0xFB | (4 * v73);
          }
          else if ( v29 == (_QWORD *)v112[0]
                 && *(_DWORD *)(v34 + 40) == *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C) )
          {
            v79 = v31;
            v87 = v14;
            v101 = v29;
            sub_85E680(*v29, (int)dword_4F04C5C);
            sub_877D50(v101, *a1);
            v75 = v101;
            v76 = v87;
            if ( v79 && !v79[1] )
            {
              v89 = v101;
              v103 = v76;
              sub_877D50(v79, *a1);
              v76 = v103;
              v75 = v89;
            }
            v88 = v76;
            v102 = v75;
            v35 = (unsigned int)dword_4F04C5C;
            sub_85E280(*v75, (unsigned int)dword_4F04C5C);
            v40 = v88;
            v39 = (__int64)v102;
          }
          else
          {
            v35 = *a1;
            v77 = v31;
            v80 = v14;
            v92 = v29;
            sub_877D50(v29, *a1);
            v39 = (__int64)v92;
            v40 = v80;
            if ( v77 && !v77[1] )
            {
              v35 = *a1;
              v90 = v92;
              v104 = v40;
              sub_877D50(v77, *a1);
              v40 = v104;
              v39 = (__int64)v90;
            }
          }
          v81 = v40;
          v93 = v39;
          sub_66A6A0(v39, v35, v36, v37, v38);
          v14 = v81;
          if ( (unsigned __int8)(*(_BYTE *)(v93 + 140) - 9) <= 2u )
          {
            sub_6434D0(v93);
            v12 = v112[0];
            v14 = v81;
            v13 = *(_BYTE *)(v112[0] + 140);
            goto LABEL_10;
          }
          v12 = v112[0];
          if ( (**(_BYTE **)(v93 + 176) & 1) != 0 )
          {
            v74 = *(_QWORD *)(v93 + 168);
            if ( (*(_BYTE *)(v93 + 161) & 0x10) != 0 )
              v74 = *(_QWORD *)(v74 + 96);
            while ( v74 )
            {
              *(_BYTE *)(v74 + 88) = *(_BYTE *)(v93 + 88) & 0x70 | *(_BYTE *)(v74 + 88) & 0x8F;
              v74 = *(_QWORD *)(v74 + 120);
            }
          }
        }
      }
    }
    goto LABEL_42;
  }
  if ( !v6 )
  {
LABEL_40:
    v14 = 0;
    goto LABEL_41;
  }
  v10 = *(_BYTE *)(v6 + 80);
  if ( v10 != 3 )
  {
LABEL_8:
    v11 = a1 + 1;
    v12 = v112[0];
    v108 = a1 + 1;
    if ( dword_4F077C4 != 2 || (unsigned __int8)(v10 - 4) > 2u )
    {
      v13 = *(_BYTE *)(v112[0] + 140);
      v14 = 0;
      goto LABEL_10;
    }
    v25 = *(_QWORD *)(v9 + 88);
    if ( v25 == v112[0] )
    {
      v26 = 1;
      goto LABEL_102;
    }
    goto LABEL_85;
  }
  v25 = *(_QWORD *)(v9 + 88);
LABEL_44:
  v12 = v112[0];
  if ( v112[0] == v25 )
  {
    v26 = 1;
    v11 = a1 + 1;
    goto LABEL_46;
  }
  v11 = a1 + 1;
LABEL_85:
  v82 = v11;
  v94 = a1[3];
  v109 = v25;
  v41 = sub_8D97D0(v25, v12, 0, v25, v7);
  v25 = v109;
  v8 = v94;
  v11 = v82;
  if ( !v41 )
  {
    v43 = *(_QWORD *)(v9 + 72);
    v44 = *(unsigned __int8 *)(v9 + 80);
    if ( v43 && *(char *)(v9 + 81) >= 0 )
    {
      if ( (_BYTE)v44 != 3 )
        goto LABEL_89;
    }
    else
    {
      v43 = 0;
      if ( (_BYTE)v44 != 3 )
        goto LABEL_91;
    }
    if ( dword_4F04C44 != -1
      || (v58 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v58 + 6) & 6) != 0)
      || *(_BYTE *)(v58 + 4) == 12 )
    {
      v78 = v112[0];
      v59 = sub_8DBE70(v109);
      v25 = v109;
      v8 = v94;
      v11 = v82;
      if ( v59 )
        goto LABEL_145;
      v60 = sub_8DBE70(v78);
      v25 = v109;
      v8 = v94;
      v11 = v82;
      if ( v60 )
        goto LABEL_145;
    }
    if ( !v43 )
      goto LABEL_91;
LABEL_89:
    v45 = *(_QWORD *)(v43 + 88);
    if ( v45 != v112[0] )
    {
      v83 = v11;
      v95 = v8;
      v110 = v25;
      v46 = sub_8D97D0(v112[0], v45, 0, v25, v42);
      v25 = v110;
      v8 = v95;
      v11 = v83;
      if ( !v46 )
      {
LABEL_91:
        v47 = *(_BYTE *)(v25 + 140);
        if ( v47 == 12 )
        {
          v48 = v25;
          do
          {
            v48 = *(_QWORD *)(v48 + 160);
            v47 = *(_BYTE *)(v48 + 140);
          }
          while ( v47 == 12 );
        }
        v108 = v11;
        if ( v47 )
        {
          v12 = v112[0];
          v14 = dword_4F077C0;
          if ( !dword_4F077C0 )
          {
LABEL_42:
            v13 = *(_BYTE *)(v12 + 140);
            goto LABEL_10;
          }
          v96 = v25;
          v14 = sub_8D2870(v112[0]);
          if ( !v14 )
          {
LABEL_97:
            v12 = v112[0];
            goto LABEL_42;
          }
          v67 = sub_8D2780(v96);
          v12 = v112[0];
          v68 = v96;
          v14 = v67;
          v69 = v67 == 0;
          v13 = *(_BYTE *)(v112[0] + 140);
          if ( !v69 )
          {
            v70 = v112[0];
            if ( v13 == 12 )
            {
              do
                v70 = *(_QWORD *)(v70 + 160);
              while ( *(_BYTE *)(v70 + 140) == 12 );
            }
            v71 = *(_BYTE *)(v70 + 160);
            if ( *(_BYTE *)(v96 + 140) == 12 )
            {
              do
                v68 = *(_QWORD *)(v68 + 160);
              while ( *(_BYTE *)(v68 + 140) == 12 );
            }
            v14 = 0;
            if ( v71 == *(_BYTE *)(v68 + 160) )
            {
              v14 = sub_729F80(*(unsigned int *)(v9 + 48)) != 0;
              goto LABEL_97;
            }
          }
LABEL_10:
          if ( v13 == 12 && *(_BYTE *)(v12 + 184) == 8 && !*(_QWORD *)(v12 + 104) )
          {
            v28 = *(_QWORD *)(v12 + 160);
            *(_BYTE *)(v12 + 184) = 0;
            v112[0] = v28;
          }
          else
          {
            v91 = v14;
            v15 = sub_7259C0(12);
            v14 = v91;
            v12 = v15;
            *(_QWORD *)(v15 + 160) = v112[0];
          }
          if ( (*(_BYTE *)(a2 + 126) & 4) != 0 )
            *(_BYTE *)(v12 + 90) |= 0x40u;
          v9 = sub_886420(v12, a1, (unsigned int)dword_4F04C5C, v14);
          sub_877D80(v12, v9);
          *(_BYTE *)(a2 + 127) |= 0x10u;
          if ( dword_4F077C4 == 2 )
          {
            if ( a3 )
            {
              sub_877E20(v9, v12, a3);
              *(_BYTE *)(v12 + 88) = *(_BYTE *)(v106 + 5) & 3 | *(_BYTE *)(v12 + 88) & 0xFC;
            }
            else
            {
              sub_877E90(v9, v12);
              v52 = *(_QWORD *)(v12 + 40);
              if ( v52 && *(_BYTE *)(v52 + 28) == 3 )
              {
                v16 = *(_QWORD *)(v52 + 32);
                goto LABEL_17;
              }
            }
          }
          v16 = 0;
LABEL_17:
          sub_8756F0(3, v9, v108, *(_QWORD *)(a2 + 352));
          v17 = *(_QWORD *)(a2 + 352);
          if ( v17 && !*(_BYTE *)(v17 + 16) )
          {
            v27 = *(_QWORD **)(v17 + 8);
            if ( v27 )
            {
              *(_QWORD *)(a2 + 352) = *v27;
              if ( (*(_BYTE *)(v12 + 89) & 5) != 1 )
                goto LABEL_20;
              goto LABEL_53;
            }
            *(_QWORD *)(a2 + 352) = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 328);
          }
          if ( (*(_BYTE *)(v12 + 89) & 5) != 1 )
            goto LABEL_20;
LABEL_53:
          sub_85E280(v9, (unsigned int)dword_4F04C5C);
LABEL_20:
          sub_729470(v12, a4);
          sub_7365B0(v12, (unsigned int)dword_4F04C5C);
          v18 = v112[0];
          v19 = *(_BYTE *)(v112[0] + 140);
          if ( v19 == 12 )
          {
            v20 = v112[0];
            do
            {
              v20 = *(_QWORD *)(v20 + 160);
              v19 = *(_BYTE *)(v20 + 140);
            }
            while ( v19 == 12 );
          }
          if ( !v19 )
            goto LABEL_27;
          if ( dword_4F04C5C )
          {
            if ( !v16
              || v16 != *(_QWORD *)(unk_4D049B8 + 88LL)
              || strcmp(*(const char **)(*(_QWORD *)v9 + 8LL), "size_t") )
            {
              goto LABEL_27;
            }
          }
          else
          {
            if ( strcmp(*(const char **)(*(_QWORD *)v9 + 8LL), "size_t") )
              goto LABEL_27;
            sub_7604D0(v12, 6);
            *(_BYTE *)(v12 + 141) |= 1u;
            v18 = v112[0];
          }
          if ( (unsigned int)sub_8D2780(v18) )
          {
            v54 = *(_BYTE *)(v112[0] + 140);
            v55 = unk_4F06A51;
            if ( v54 == 12 )
            {
              v56 = v112[0];
              do
                v56 = *(_QWORD *)(v56 + 160);
              while ( *(_BYTE *)(v56 + 140) == 12 );
              if ( *(_BYTE *)(v56 + 160) != unk_4F06A51 )
                goto LABEL_132;
LABEL_174:
              if ( (unsigned int)sub_8D4C10(v112[0], dword_4F077C4 != 2) )
              {
                v55 = unk_4F06A51;
                goto LABEL_132;
              }
LABEL_27:
              if ( unk_4D047EC )
              {
                if ( unk_4F04C50 )
                {
                  if ( (unsigned int)sub_8DD010(v112[0]) )
                  {
                    v53 = sub_86E480(22, v108);
                    *(_BYTE *)(v53 + 72) = 1;
                    *(_QWORD *)(v53 + 80) = v12;
                    *(_BYTE *)(v12 + 186) |= 1u;
                    if ( unk_4D047E8 )
                    {
                      if ( (unsigned int)sub_86D9F0() )
                        sub_6851C0(1233, v108);
                    }
                  }
                }
              }
              *(_QWORD *)a2 = v9;
              sub_644920((_QWORD *)a2, 1);
              v21 = *(_BYTE *)(v112[0] + 140);
              if ( v21 == 12 )
              {
                v22 = v112[0];
                do
                {
                  v22 = *(_QWORD *)(v22 + 160);
                  v21 = *(_BYTE *)(v22 + 140);
                }
                while ( v21 == 12 );
              }
              if ( v21 && *(char *)(v12 + 90) >= 0 )
                sub_8D9350(v112[0], v108);
              return sub_854980(v9, 0);
            }
            if ( *(_BYTE *)(v112[0] + 160) == unk_4F06A51 )
            {
              if ( (v54 & 0xFB) != 8 )
                goto LABEL_27;
              goto LABEL_174;
            }
          }
          else
          {
            v55 = unk_4F06A51;
          }
LABEL_132:
          v57 = sub_72BA30(v55);
          sub_685330(867, v108, v57);
          goto LABEL_27;
        }
      }
    }
LABEL_145:
    v26 = 0;
    goto LABEL_46;
  }
  v26 = 1;
LABEL_46:
  v108 = v11;
  if ( dword_4F077C4 != 2 )
    goto LABEL_47;
  v12 = v25;
LABEL_102:
  v49 = *(_BYTE *)(v12 + 140);
  if ( v49 == 12 )
  {
    v50 = v12;
    do
    {
      v50 = *(_QWORD *)(v50 + 160);
      v49 = *(_BYTE *)(v50 + 140);
    }
    while ( v49 == 12 );
  }
  v108 = v11;
  if ( v49 && (unsigned __int8)(*(_BYTE *)(v9 + 80) - 4) > 2u && a3 && *(_BYTE *)(v8 + 80) != 16 )
  {
    if ( v12 == a3 || dword_4F07588 && (v51 = *(_QWORD *)(a3 + 32), *(_QWORD *)(v12 + 32) == v51) && v51 )
    {
      v85 = v26;
      v99 = v11;
      sub_684AA0(dword_4D04964 == 0 ? 5 : 8, 280, v11);
      v11 = v99;
      v26 = v85;
    }
    else if ( dword_4D04964 )
    {
      v86 = v26;
      v100 = v11;
      sub_6851C0(1307, v11);
      v11 = v100;
      v26 = v86;
    }
    else
    {
      v84 = v26;
      v97 = v11;
      if ( (*(_BYTE *)(v12 + 88) & 3) == (*(_BYTE *)(v106 + 5) & 3) )
      {
        sub_684B30(1307, v11);
        v26 = v84;
        v11 = v97;
      }
      else
      {
        sub_685490(720, v11, v9);
        v11 = v97;
        v26 = v84;
      }
    }
  }
LABEL_47:
  if ( *(_BYTE *)(a1[3] + 80LL) == 24 )
  {
    v12 = v112[0];
    if ( (*((_BYTE *)a1 + 17) & 0x40) == 0 )
    {
      *((_BYTE *)a1 + 16) &= ~0x80u;
      v14 = 1;
      a1[3] = 0;
      v13 = *(_BYTE *)(v12 + 140);
      goto LABEL_10;
    }
    goto LABEL_50;
  }
  if ( *(_BYTE *)(v9 + 80) != 3 )
  {
    v12 = v112[0];
LABEL_50:
    v13 = *(_BYTE *)(v12 + 140);
    v14 = 1;
    goto LABEL_10;
  }
  if ( dword_4F077C4 != 2 )
  {
    if ( unk_4F07778 <= 201111 )
    {
      v61 = v26 == 0 ? 983 : 301;
      goto LABEL_149;
    }
    v61 = 983;
    if ( !v26 )
    {
LABEL_149:
      v62 = 5;
      if ( dword_4D04964 )
        v62 = byte_4F07472[0];
      v111 = v11;
      sub_684AA0(v62, v61, v11);
      v11 = v111;
    }
  }
  sub_8756F0(1, v9, v11, *(_QWORD *)(a2 + 352));
  v63 = *(_QWORD *)(a2 + 352);
  if ( v63 && !*(_BYTE *)(v63 + 16) )
  {
    v64 = *(__int64 **)(v63 + 8);
    if ( v64 )
      v65 = *v64;
    else
      v65 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 328);
    *(_QWORD *)(a2 + 352) = v65;
  }
  v66 = v112[0];
  if ( *(_BYTE *)(v112[0] + 140) == 12 && *(_BYTE *)(v112[0] + 184) == 8 && !*(_QWORD *)(v112[0] + 104) )
    v66 = *(_QWORD *)(v112[0] + 160);
  sub_86A3D0(*(_QWORD *)(v9 + 88), v66, 0, (unsigned __int8)((*(_BYTE *)(a2 + 126) & 4) != 0) << 6, a4);
  *(_QWORD *)a2 = v9;
  sub_644920((_QWORD *)a2, 0);
  return sub_854980(v9, 0);
}
