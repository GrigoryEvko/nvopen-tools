// Function: sub_6CF140
// Address: 0x6cf140
//
__int64 __fastcall sub_6CF140(__int64 a1, __int64 *a2, _DWORD *a3, __int64 *a4, __int64 a5)
{
  __int64 *v6; // r15
  __int64 v7; // r8
  int v8; // edx
  __int64 v9; // rax
  __int64 result; // rax
  __int64 *v11; // rsi
  char v12; // dl
  __int64 v13; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdi
  unsigned int v20; // eax
  int v21; // r8d
  __int64 v22; // rax
  __int64 v23; // r11
  unsigned __int8 v24; // al
  __int64 v25; // rax
  char i; // dl
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // r11
  __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  unsigned int v38; // eax
  __int64 v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // rax
  unsigned int v44; // r14d
  __int64 v45; // rax
  __int64 v46; // rax
  char j; // dl
  int v48; // eax
  __int64 v49; // rax
  int v50; // eax
  __int64 v51; // [rsp-10h] [rbp-360h]
  __int64 v52; // [rsp-8h] [rbp-358h]
  __int64 v53; // [rsp+8h] [rbp-348h]
  _BOOL4 v54; // [rsp+10h] [rbp-340h]
  __int64 v55; // [rsp+10h] [rbp-340h]
  __int64 v56; // [rsp+10h] [rbp-340h]
  int v57; // [rsp+18h] [rbp-338h]
  __int64 v58; // [rsp+18h] [rbp-338h]
  __int64 v59; // [rsp+18h] [rbp-338h]
  unsigned __int16 v60; // [rsp+28h] [rbp-328h]
  char v61; // [rsp+3Bh] [rbp-315h] BYREF
  unsigned int v62; // [rsp+3Ch] [rbp-314h] BYREF
  _BOOL4 v63; // [rsp+40h] [rbp-310h] BYREF
  unsigned int v64; // [rsp+44h] [rbp-30Ch] BYREF
  __int64 v65; // [rsp+48h] [rbp-308h] BYREF
  __int64 v66; // [rsp+50h] [rbp-300h] BYREF
  __int64 v67; // [rsp+58h] [rbp-2F8h] BYREF
  _BYTE v68[352]; // [rsp+60h] [rbp-2F0h] BYREF
  _QWORD v69[2]; // [rsp+1C0h] [rbp-190h] BYREF
  char v70; // [rsp+1D0h] [rbp-180h]
  int v71; // [rsp+204h] [rbp-14Ch] BYREF
  __int64 v72; // [rsp+20Ch] [rbp-144h]
  _BYTE v73[256]; // [rsp+250h] [rbp-100h] BYREF

  v61 = 120;
  v63 = 0;
  *a3 = 0;
  if ( a2 )
  {
    v6 = (__int64 *)v68;
    v60 = *((_WORD *)a2 + 4);
    sub_6F8AB0((_DWORD)a2, (unsigned int)v68, (unsigned int)v69, 0, (unsigned int)&v67, (unsigned int)&v62, 0);
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
      goto LABEL_3;
    if ( (unsigned int)sub_6E5430(a2, v68, v51, v52, v7) )
      sub_6851C0(0x39u, &v67);
LABEL_29:
    sub_6E6260(a4);
    sub_6E6450(v6);
    sub_6E6450(v69);
    goto LABEL_5;
  }
  v6 = (__int64 *)a1;
  v60 = word_4F06418[0];
  v67 = *(_QWORD *)&dword_4F063F8;
  v62 = dword_4F06650[0];
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_6E5430(a1, 0, a3, a4, a5) )
    {
      a1 = 57;
      a2 = &v67;
      sub_6851C0(0x39u, &v67);
    }
    v57 = 1;
  }
  else
  {
    v57 = 0;
  }
  sub_7B8B50(a1, a2, a3, a4);
  if ( word_4F06418[0] == 73 && dword_4D04428 )
  {
    v43 = sub_6BA760(0, 0);
    sub_6E9FE0(v43, v69);
    *a3 = 1;
  }
  else
  {
    sub_69ED20((__int64)v69, 0, 2, 0);
  }
  if ( v57 )
    goto LABEL_29;
LABEL_3:
  if ( dword_4F077C4 == 2 && ((unsigned int)sub_68FE10(v6, 1, 1) || (unsigned int)sub_68FE10(v69, 0, 1)) )
  {
    v21 = sub_8D2870(*v6);
    if ( v21 )
      v21 = v70 != 5;
    sub_84EC30(
      byte_4B6D300[v60],
      0,
      0,
      1,
      v21,
      (_DWORD)v6,
      (__int64)v69,
      (__int64)&v67,
      v62,
      0,
      0,
      (__int64)a4,
      0,
      0,
      (__int64)&v63);
  }
  if ( !v63 )
  {
    if ( (unsigned int)sub_6E9250(&v67) )
      goto LABEL_20;
    v54 = v63;
    if ( v63 )
      goto LABEL_5;
    sub_6F69D0(v6, 4);
    if ( dword_4F077C4 == 2 )
    {
      v39 = *v6;
      if ( (unsigned int)sub_8D2870(*v6) )
      {
        if ( unk_4D04950 )
        {
          sub_6E5C80(unk_4F07470, 428, (char *)v6 + 68);
        }
        else
        {
          if ( (unsigned int)sub_6E5430(v39, 4, v40, v41, v42) )
            sub_6851C0(0x1FFu, &v67);
          sub_6E6840(v6);
        }
      }
    }
    if ( (unsigned int)sub_702F90(v6) )
      sub_6ECF90(v6, 1);
    v11 = 0;
    sub_6F69D0(v69, 0);
    switch ( v60 )
    {
      case '9':
      case ':':
        if ( HIDWORD(qword_4F077B4) && (unsigned int)sub_6FD310(v60, v6, v69, &v67, &v65, &v61) )
          goto LABEL_54;
        sub_6E9580(v6);
        sub_6E9580(v69);
        goto LABEL_14;
      case ';':
        goto LABEL_13;
      case '<':
        if ( dword_4F077C4 != 2 )
          goto LABEL_35;
        if ( !(unsigned int)sub_8D29A0(*v6) )
          goto LABEL_35;
        v34 = v69[0];
        if ( !(unsigned int)sub_8D2E30(v69[0]) )
          goto LABEL_35;
        v38 = sub_6E9530(v34, 0, v35, v36, v37);
        sub_702E30(v69, v38);
        v54 = 1;
        goto LABEL_14;
      case '=':
LABEL_35:
        if ( HIDWORD(qword_4F077B4) )
        {
          v11 = v6;
          if ( (unsigned int)sub_6FD310(v60, v6, v69, &v67, &v65, &v61) )
            goto LABEL_54;
        }
        if ( (unsigned int)sub_8D2D80(*v6) )
        {
          sub_6E9580(v69);
          goto LABEL_14;
        }
        v18 = dword_4F077C0;
        if ( (dword_4F077C0
           || (v19 = dword_4F077BC) != 0
           && (v11 = (__int64 *)(unsigned int)qword_4F077B4, !(_DWORD)qword_4F077B4)
           && qword_4F077A8 > 0x9DCFu)
          && (v19 = *v6, (unsigned int)sub_8D2E30(*v6))
          && ((v45 = sub_8D46C0(*v6), (unsigned int)sub_8D2600(v45))
           || (v19 = sub_8D46C0(*v6), (unsigned int)sub_8D2310(v19))) )
        {
          if ( (unsigned int)sub_6E9350(v69) )
          {
            if ( *(char *)(qword_4D03C50 + 20LL) >= 0 && (unsigned int)sub_6E53E0(5, 1143, &v67) )
              sub_684B30(0x477u, &v67);
            v54 = 1;
LABEL_14:
            if ( *((_BYTE *)v6 + 16) )
            {
              v12 = *(_BYTE *)(*v6 + 140);
              if ( v12 == 12 )
              {
                v13 = *v6;
                do
                {
                  v13 = *(_QWORD *)(v13 + 160);
                  v12 = *(_BYTE *)(v13 + 140);
                }
                while ( v12 == 12 );
              }
              if ( v12 && v70 )
              {
                v25 = v69[0];
                for ( i = *(_BYTE *)(v69[0] + 140LL); i == 12; i = *(_BYTE *)(v25 + 140) )
                  v25 = *(_QWORD *)(v25 + 160);
                if ( i )
                {
                  v53 = *v6;
                  v27 = sub_73D720(*v6);
                  v23 = v53;
                  v65 = v27;
                  if ( v54 )
                  {
                    v66 = *v6;
                    v48 = sub_8D29A0(v66);
                    v33 = v53;
                    if ( v48 )
                    {
                      v32 = v69[0];
                      v66 = v69[0];
                    }
                    else
                    {
                      v32 = v66;
                    }
                    goto LABEL_80;
                  }
                  v30 = (unsigned int)dword_4F077C4;
                  if ( (unsigned __int16)(v60 - 62) <= 1u )
                  {
                    v55 = v53;
                    if ( dword_4F077C4 != 1 )
                    {
                      v66 = *v6;
                      sub_6FC420(v69);
                      v32 = v66;
                      v33 = v53;
                      goto LABEL_80;
                    }
                    v66 = sub_6E8B10(v6, v69, (unsigned int)dword_4F077C4, v28, v29);
                    v31 = sub_72BA30(5);
                  }
                  else
                  {
                    if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
                    {
                      v50 = sub_6FCD00(v60, v6, v69, &v67, &v66, &v61);
                      v23 = v53;
                      if ( v50 )
                      {
                        v66 = 0;
                        v24 = v61;
                        goto LABEL_55;
                      }
                    }
                    v55 = v23;
                    v66 = sub_6E8B10(v6, v69, v30, v28, v29);
                    v31 = v66;
                  }
                  sub_6FC3F0(v31, v69, 1);
                  v32 = v66;
                  v33 = v55;
LABEL_80:
                  v56 = v33;
                  v24 = sub_6E9930(v60, v32);
                  v23 = v56;
                  v61 = v24;
LABEL_55:
                  v59 = v23;
                  sub_6F7CB0(v6, v69, v24, v65, a4);
                  if ( dword_4F077C4 == 2 )
                  {
                    if ( !*((_BYTE *)a4 + 16) )
                      goto LABEL_111;
                    v46 = *a4;
                    for ( j = *(_BYTE *)(*a4 + 140); j == 12; j = *(_BYTE *)(v46 + 140) )
                      v46 = *(_QWORD *)(v46 + 160);
                    if ( j )
                    {
                      v49 = a4[18];
                      *(_BYTE *)(v49 + 25) |= 1u;
                      *(_BYTE *)(v49 + 58) |= 1u;
                      *a4 = v59;
                      *(_QWORD *)v49 = v59;
                      a4[11] = v6[11];
                      sub_6E6A20(a4);
                    }
                    else
                    {
LABEL_111:
                      sub_6E6870(a4);
                    }
                  }
                  if ( (unsigned __int16)(v60 - 58) <= 1u )
                  {
                    if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 && (unsigned int)sub_6E97C0(v69) )
                    {
                      v44 = 179;
                      if ( v60 == 58 )
                        v44 = 39;
                      if ( (unsigned int)sub_6E53E0(5, v44, &v71) )
                        sub_684B30(v44, &v71);
                    }
                  }
                  else if ( (unsigned __int16)(v60 - 62) <= 1u
                         && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0
                         && v70 == 2
                         && v73[173] == 1 )
                  {
                    sub_7131E0(v73, *v6, &v64);
                    if ( v64 )
                      sub_69D070(v64, &v71);
                  }
                  break;
                }
              }
            }
LABEL_20:
            sub_6E6260(a4);
            break;
          }
        }
        else
        {
          v20 = sub_6E9530(v19, v11, v16, v17, v18);
          if ( (unsigned int)sub_702E30(v6, v20) )
          {
            v54 = sub_6E9350(v69) != 0;
            goto LABEL_14;
          }
        }
        v54 = 0;
        goto LABEL_14;
      case '>':
      case '?':
        if ( HIDWORD(qword_4F077B4) )
        {
          if ( !(_DWORD)qword_4F077B4 )
          {
            if ( qword_4F077A8 <= 0x9F5Fu )
              goto LABEL_67;
            goto LABEL_66;
          }
        }
        else if ( !(_DWORD)qword_4F077B4 )
        {
          goto LABEL_67;
        }
        if ( qword_4F077A0 <= 0x9C3Fu )
          goto LABEL_67;
LABEL_66:
        if ( (unsigned int)sub_6FD310(v60, v6, v69, &v67, &v65, &v61) )
        {
LABEL_54:
          v58 = *v6;
          v22 = sub_73D720(v65);
          v23 = v58;
          v66 = v22;
          v24 = v61;
          goto LABEL_55;
        }
LABEL_67:
        sub_6E93E0(v6);
        sub_6E9350(v69);
        goto LABEL_14;
      case '@':
      case 'A':
      case 'B':
        if ( HIDWORD(qword_4F077B4) && (unsigned int)sub_6FD310(v60, v6, v69, &v67, &v65, &v61) )
          goto LABEL_54;
LABEL_13:
        sub_6E9350(v6);
        sub_6E9350(v69);
        goto LABEL_14;
      default:
        sub_721090(v69);
    }
  }
LABEL_5:
  v8 = *((_DWORD *)v6 + 17);
  *((_WORD *)a4 + 36) = *((_WORD *)v6 + 36);
  *((_DWORD *)a4 + 17) = v8;
  *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)a4 + 68);
  v9 = v72;
  *(__int64 *)((char *)a4 + 76) = v72;
  *(_QWORD *)&dword_4F061D8 = v9;
  sub_6E3280(a4, &v67);
  sub_6E3BA0(a4, &v67, v62, 0);
  result = sub_6E26D0(2, a4);
  if ( dword_4F077C4 == 2 )
  {
    result = *(_BYTE *)(*a4 + 140) & 0xFB;
    if ( (*(_BYTE *)(*a4 + 140) & 0xFB) == 8 )
    {
      result = sub_8D4C10(*a4, 0);
      if ( (result & 2) != 0 )
      {
        v15 = 4;
        if ( dword_4F077C4 == 2 )
          v15 = (unsigned int)(unk_4F07778 > 202001) + 4;
        return sub_6E5C80(v15, 3012, (char *)a4 + 68);
      }
    }
  }
  return result;
}
