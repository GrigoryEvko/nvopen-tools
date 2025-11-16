// Function: sub_6B3BD0
// Address: 0x6b3bd0
//
char __fastcall sub_6B3BD0(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  char v6; // dl
  _QWORD *v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // r13
  char v15; // al
  _BYTE *v16; // rax
  unsigned int v17; // esi
  char v18; // dl
  bool v19; // r15
  int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // r15
  int v23; // edx
  __int64 v24; // rax
  char result; // al
  unsigned __int8 v26; // al
  _BYTE *v27; // rax
  unsigned int v28; // r15d
  unsigned int v29; // ecx
  char v30; // dl
  unsigned __int16 v31; // di
  __int64 v32; // rdx
  unsigned int v33; // r11d
  char v34; // al
  __int64 v35; // rax
  unsigned int v36; // esi
  int v37; // eax
  int v38; // eax
  unsigned int v39; // edi
  bool v40; // [rsp+7h] [rbp-4B9h]
  __int64 v41; // [rsp+8h] [rbp-4B8h]
  int v42; // [rsp+10h] [rbp-4B0h]
  unsigned int v43; // [rsp+14h] [rbp-4ACh]
  unsigned int v45; // [rsp+28h] [rbp-498h]
  bool v46; // [rsp+2Ch] [rbp-494h]
  bool v47; // [rsp+2Dh] [rbp-493h]
  unsigned __int16 v48; // [rsp+2Eh] [rbp-492h]
  char v49; // [rsp+38h] [rbp-488h]
  unsigned int v50; // [rsp+48h] [rbp-478h]
  unsigned int v51; // [rsp+4Ch] [rbp-474h]
  unsigned __int8 v52; // [rsp+57h] [rbp-469h] BYREF
  unsigned int v53; // [rsp+58h] [rbp-468h] BYREF
  int v54; // [rsp+5Ch] [rbp-464h] BYREF
  __int64 v55; // [rsp+60h] [rbp-460h] BYREF
  __int64 v56; // [rsp+68h] [rbp-458h] BYREF
  _BYTE v57[352]; // [rsp+70h] [rbp-450h] BYREF
  _QWORD v58[2]; // [rsp+1D0h] [rbp-2F0h] BYREF
  char v59; // [rsp+1E0h] [rbp-2E0h]
  char v60; // [rsp+210h] [rbp-2B0h]
  __int64 v61; // [rsp+21Ch] [rbp-2A4h]
  _QWORD v62[50]; // [rsp+330h] [rbp-190h] BYREF

  v54 = 0;
  v6 = *(_BYTE *)(qword_4D03C50 + 17LL);
  v49 = v6 & 1;
  v47 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x10) != 0;
  v51 = v6 & 1;
  v46 = (v6 & 4) != 0;
  v50 = *(_BYTE *)(qword_4D03C50 + 21LL) >> 7;
  if ( a2 )
  {
    v7 = v57;
    v48 = *(_WORD *)(a2 + 8);
    sub_6F8AB0(a2, (unsigned int)v57, (unsigned int)v58, 0, (unsigned int)&v56, (unsigned int)&v53, 0);
  }
  else
  {
    v7 = a1;
    v48 = word_4F06418[0];
    v56 = *(_QWORD *)&dword_4F063F8;
    v53 = dword_4F06650[0];
  }
  sub_6E16F0(v7[11], 0);
  sub_6E17F0(v7);
  if ( dword_4F077C4 == 2 )
  {
    v26 = *(_BYTE *)(qword_4D03C50 + 16LL);
    if ( v26 )
    {
      if ( v26 > 3u || word_4D04898 )
      {
        v10 = (_QWORD *)byte_4B6D300[v48];
        if ( (unsigned int)sub_7D3880(v10) )
        {
          if ( *((_BYTE *)v7 + 17) == 1 && v49 )
          {
            v10 = v7;
            sub_6ECC10(v7, 0, v8);
            v42 = 1;
            v43 = 0;
          }
          else
          {
            v42 = 1;
            v43 = 0;
          }
LABEL_16:
          v11 = v50;
          if ( v50 )
            goto LABEL_17;
          goto LABEL_9;
        }
      }
    }
  }
  v10 = (_QWORD *)v51;
  if ( !v51
    || (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0
    || dword_4F077C4 == 2 && (v10 = (_QWORD *)*v7, (unsigned int)sub_8D3A70(*v7)) )
  {
    v42 = 0;
    v43 = 0;
    goto LABEL_16;
  }
  v10 = v7;
  sub_6F69D0(v7, 0);
  if ( *((_BYTE *)v7 + 16) != 2 )
    goto LABEL_8;
  v10 = (_QWORD *)*v7;
  if ( !(unsigned int)sub_8D3350(*v7) )
    goto LABEL_8;
  v10 = v7 + 18;
  if ( !(unsigned int)sub_70FCE0(v7 + 18) )
    goto LABEL_8;
  v10 = v7;
  v37 = sub_6E9820(v7);
  if ( v48 != 52 || !v37 )
  {
    if ( v48 == 53 && !v37 )
    {
      if ( a2 )
      {
        v42 = 0;
        v41 = 1;
        v43 = v51;
        v45 = v51;
        v51 = 0;
        goto LABEL_19;
      }
      v41 = 1;
      v28 = 4;
      goto LABEL_114;
    }
LABEL_8:
    v11 = v50;
    v42 = 0;
    v43 = v51;
    if ( v50 )
    {
LABEL_17:
      if ( a2 )
      {
        v45 = 0;
        v41 = 0;
        goto LABEL_19;
      }
      sub_7B8B50(v10, v11, v8, v9);
      v27 = (_BYTE *)qword_4D03C50;
      v41 = 0;
      v28 = (v48 == 52) + 4;
      v29 = word_4D04898;
      v30 = v49 | *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFE;
      *(_BYTE *)(qword_4D03C50 + 17LL) = v30;
      v45 = v29;
      if ( v29 )
      {
        if ( v27[16] <= 3u && v42 )
        {
          v45 = 0;
          v27[17] = v30 | 4;
          v42 = 1;
        }
        else
        {
          v45 = 0;
          v41 = 0;
        }
      }
      goto LABEL_64;
    }
LABEL_9:
    v10 = (_QWORD *)*v7;
    if ( (unsigned int)sub_8D3350(*v7) && (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) == 0 )
    {
      v14 = v7 + 18;
      v62[0] = sub_724DC0(v10, v11, v8, v9, v12, v13);
      v11 = v62[0];
      v15 = *((_BYTE *)v7 + 16);
      if ( v15 == 2 || v15 == 1 && (unsigned int)sub_719770(v7[18], v62[0], 0, 1) && (v14 = (_QWORD *)v62[0]) != 0 )
      {
        if ( (unsigned int)sub_70FCE0(v14) )
        {
          if ( (v38 = sub_711520(v14, v11), v48 == 52) && v38 || v48 == 53 && !v38 )
            *(_BYTE *)(qword_4D03C50 + 21LL) |= 0x80u;
        }
      }
      v10 = v62;
      sub_724E30(v62);
    }
    goto LABEL_17;
  }
  if ( a2 )
  {
    v42 = 0;
    v41 = 0;
    v43 = v51;
    v45 = v51;
    v51 = 0;
    goto LABEL_19;
  }
  v41 = 0;
  v28 = 5;
LABEL_114:
  sub_7B8B50(v7, 0, v8, v9);
  v39 = v51;
  v27 = (_BYTE *)qword_4D03C50;
  v42 = 0;
  v51 = 0;
  v45 = v39;
  v43 = v39;
  *(_BYTE *)(qword_4D03C50 + 17LL) &= ~1u;
LABEL_64:
  v40 = (v27[19] & 0x20) != 0;
  v27[18] |= 0x10u;
  if ( (a3 & 0x2000) != 0 )
  {
    v31 = word_4F06418[0];
    if ( !sub_6878E0(word_4F06418[0]) && (!sub_687960(v31) || dword_4D04964) )
      sub_6851C0(0xBDDu, &dword_4F063F8);
  }
  sub_69ED20((__int64)v58, 0, v28, a3);
  v32 = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 18LL) = (16 * v47) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0xEF;
  v33 = word_4D04898;
  v34 = v49 | *(_BYTE *)(v32 + 17) & 0xFE;
  *(_BYTE *)(v32 + 17) = v34;
  if ( v33 )
  {
    *(_BYTE *)(v32 + 17) = (4 * v46) | v34 & 0xFB;
    *(_BYTE *)(v32 + 19) = *(_BYTE *)(v32 + 19) & 0xDF | (32 * v40);
    if ( dword_4F077C4 != 2 )
      goto LABEL_20;
    goto LABEL_71;
  }
LABEL_19:
  if ( dword_4F077C4 != 2 )
    goto LABEL_20;
LABEL_71:
  if ( (unsigned int)sub_68FE10(v7, 1, 1) || (unsigned int)sub_68FE10(v58, 0, 1) )
    sub_84EC30(
      byte_4B6D300[v48],
      0,
      0,
      1,
      0,
      (_DWORD)v7,
      (__int64)v58,
      (__int64)&v56,
      v53,
      0,
      0,
      a4,
      0,
      0,
      (__int64)&v54);
LABEL_20:
  if ( v54 )
    goto LABEL_45;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) == 2 )
  {
    sub_68BB70(v7, v58, &v56, a4, &v54);
    if ( v54 )
      goto LABEL_45;
    if ( v43 )
      goto LABEL_23;
  }
  else if ( v43 )
  {
    goto LABEL_23;
  }
  sub_6F69D0(v7, 0);
LABEL_23:
  if ( !(unsigned int)sub_8D2B80(*v7) || HIDWORD(qword_4F077B4) && dword_4F077C4 != 2 )
    sub_6FC8A0(v7);
  v16 = (_BYTE *)qword_4D03C50;
  v17 = word_4D04898;
  v18 = v51 | *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFE;
  *(_BYTE *)(qword_4D03C50 + 17LL) = v18;
  if ( v17 && v16[16] <= 3u && v42 )
    v16[17] = v18 | 4;
  v16[18] |= 0x10u;
  v19 = (v16[19] & 0x20) != 0;
  sub_6F69D0(v58, 0);
  if ( !(unsigned int)sub_8D2B80(v58[0]) )
    goto LABEL_73;
  v20 = HIDWORD(qword_4F077B4);
  if ( !HIDWORD(qword_4F077B4) )
  {
    v21 = qword_4D03C50;
    *(_BYTE *)(qword_4D03C50 + 18LL) = (16 * v47) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0xEF;
    v36 = word_4D04898;
    *(_BYTE *)(v21 + 17) = v49 | *(_BYTE *)(v21 + 17) & 0xFE;
    if ( !v36 )
      goto LABEL_36;
    goto LABEL_33;
  }
  if ( dword_4F077C4 != 2 )
  {
LABEL_73:
    sub_6FC8A0(v58);
    v20 = HIDWORD(qword_4F077B4);
    v21 = qword_4D03C50;
    *(_BYTE *)(qword_4D03C50 + 18LL) = (16 * v47) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0xEF;
    *(_BYTE *)(v21 + 17) = v49 | *(_BYTE *)(v21 + 17) & 0xFE;
    if ( !word_4D04898 )
    {
LABEL_34:
      if ( !v20 )
        goto LABEL_36;
      goto LABEL_35;
    }
LABEL_33:
    *(_BYTE *)(v21 + 17) = (4 * v46) | *(_BYTE *)(v21 + 17) & 0xFB;
    *(_BYTE *)(v21 + 19) = (32 * v19) | *(_BYTE *)(v21 + 19) & 0xDF;
    goto LABEL_34;
  }
  v21 = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 18LL) = (16 * v47) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0xEF;
  *(_BYTE *)(v21 + 17) = v49 | *(_BYTE *)(v21 + 17) & 0xFE;
  if ( word_4D04898 )
    goto LABEL_33;
LABEL_35:
  if ( dword_4F077C4 == 2 && (unsigned int)sub_6FD310(v48, v7, v58, &v56, &v55, &v52) )
  {
    v22 = sub_6E8E20(v55);
    goto LABEL_37;
  }
LABEL_36:
  v22 = sub_6EFF80();
LABEL_37:
  if ( !v45
    || v59 == 2
    || *(_BYTE *)(qword_4D03C50 + 16LL) > 3u
    && (v60 & 2) != 0
    && (dword_4D04964 || (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) == 0) )
  {
    v52 = sub_6E9930(v48, v22);
    sub_7016A0(v52, (_DWORD)v7, (unsigned int)v58, v22, a4, (unsigned int)&v56, v53);
    if ( *(_BYTE *)(a4 + 16) == 1 )
    {
      if ( dword_4F04C44 != -1
        || (v35 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v35 + 6) & 6) != 0)
        || *(_BYTE *)(v35 + 4) == 12 )
      {
        if ( (unsigned int)sub_696840((__int64)v58) )
          *(_BYTE *)(*(_QWORD *)(a4 + 144) + 27LL) |= 1u;
      }
    }
  }
  else
  {
    sub_6E7080(a4, v41);
    sub_6FC3F0(v22, a4, 1);
    *(_BYTE *)(a4 + 313) |= 4u;
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
    {
      v52 = sub_6E9930(v48, v22);
      sub_6F7CB0(v7, v58, v52, v22, v62);
      *(_QWORD *)(a4 + 288) = sub_6F6F40(v62, 0);
    }
  }
LABEL_45:
  v23 = *((_DWORD *)v7 + 17);
  *(_WORD *)(a4 + 72) = *((_WORD *)v7 + 36);
  *(_DWORD *)(a4 + 68) = v23;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a4 + 68);
  v24 = v61;
  *(_QWORD *)(a4 + 76) = v61;
  unk_4F061D8 = v24;
  sub_6E3280(a4, &v56);
  result = *((_BYTE *)v7 + 64);
  if ( dword_4D04964 | v51 )
    result |= v60;
  *(_BYTE *)(a4 + 64) = result;
  if ( !v50 )
  {
    result = qword_4D03C50;
    *(_BYTE *)(qword_4D03C50 + 21LL) &= ~0x80u;
  }
  return result;
}
