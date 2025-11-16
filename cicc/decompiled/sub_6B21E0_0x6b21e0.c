// Function: sub_6B21E0
// Address: 0x6b21e0
//
__int64 __fastcall sub_6B21E0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rbx
  _BOOL8 v6; // r14
  __int64 v7; // rax
  char i; // dl
  __int64 v9; // rdi
  char v10; // dl
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  unsigned int v15; // eax
  int v16; // edx
  __int64 v17; // rax
  unsigned int v19; // eax
  unsigned __int8 v20; // al
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rcx
  unsigned int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int8 v28; // al
  int v29; // edi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 j; // rax
  __int64 v34; // rax
  unsigned __int8 v35; // al
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int16 v39; // [rsp+8h] [rbp-328h]
  __int64 v40; // [rsp+8h] [rbp-328h]
  unsigned __int8 v41; // [rsp+1Fh] [rbp-311h] BYREF
  unsigned int v42; // [rsp+20h] [rbp-310h] BYREF
  int v43; // [rsp+24h] [rbp-30Ch] BYREF
  __int64 v44; // [rsp+28h] [rbp-308h] BYREF
  __int64 v45; // [rsp+30h] [rbp-300h] BYREF
  __int64 v46; // [rsp+38h] [rbp-2F8h] BYREF
  _BYTE v47[352]; // [rsp+40h] [rbp-2F0h] BYREF
  _QWORD v48[2]; // [rsp+1A0h] [rbp-190h] BYREF
  char v49; // [rsp+1B0h] [rbp-180h]
  __int64 v50; // [rsp+1ECh] [rbp-144h]

  v41 = 119;
  v43 = 0;
  if ( a2 )
  {
    v5 = v47;
    v39 = *(_WORD *)(a2 + 8);
    sub_6F8AB0(a2, (unsigned int)v47, (unsigned int)v48, 0, (unsigned int)&v46, (unsigned int)&v42, 0);
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
  }
  else
  {
    v5 = a1;
    v39 = word_4F06418[0];
    v46 = *(_QWORD *)&dword_4F063F8;
    v42 = dword_4F06650[0];
    sub_7B8B50(a1, 0, a3, a4);
    sub_69ED20((__int64)v48, 0, 14, 0);
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
  }
  if ( (unsigned int)sub_68FE10(v5, 1, 1) || (unsigned int)sub_68FE10(v48, 0, 1) )
    sub_84EC30(
      byte_4B6D300[v39],
      0,
      0,
      1,
      0,
      (_DWORD)v5,
      (__int64)v48,
      (__int64)&v46,
      v42,
      0,
      0,
      a3,
      0,
      0,
      (__int64)&v43);
LABEL_3:
  LODWORD(v6) = v43;
  if ( v43 )
    goto LABEL_22;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) == 2 )
  {
    sub_68BB70(v5, v48, &v46, a3, &v43);
    if ( v43 )
      goto LABEL_22;
  }
  sub_6F69D0(v5, 0);
  if ( !(unsigned int)sub_8D2D80(*v5) && (!HIDWORD(qword_4F077B4) || !(unsigned int)sub_8D2B80(*v5)) )
  {
    v19 = sub_6E9530();
    v6 = (unsigned int)sub_6FB4D0(v5, v19) != 0;
  }
  sub_6F69D0(v48, 0);
  if ( !*((_BYTE *)v5 + 16) )
    goto LABEL_21;
  v7 = *v5;
  for ( i = *(_BYTE *)(*v5 + 140LL); i == 12; i = *(_BYTE *)(v7 + 140) )
    v7 = *(_QWORD *)(v7 + 160);
  if ( !i || !v49 )
    goto LABEL_21;
  v9 = v48[0];
  v10 = *(_BYTE *)(v48[0] + 140LL);
  if ( v10 == 12 )
  {
    v11 = v48[0];
    do
    {
      v11 = *(_QWORD *)(v11 + 160);
      v10 = *(_BYTE *)(v11 + 140);
    }
    while ( v10 == 12 );
  }
  if ( !v10 )
    goto LABEL_21;
  if ( !v6 )
  {
    if ( v39 == 35 )
    {
      if ( (unsigned int)sub_8D2E30(v48[0]) && (unsigned int)sub_8D2960(*v5) )
      {
        if ( dword_4F077C0
          && ((v27 = sub_8D46C0(v48[0]), (unsigned int)sub_8D2600(v27))
           || (v34 = sub_8D46C0(v48[0]), (unsigned int)sub_8D2310(v34))) )
        {
          if ( *(char *)(qword_4D03C50 + 20LL) >= 0 )
            sub_69D070(0x477u, &v46);
        }
        else
        {
          sub_702E30(v48, 142);
        }
        v45 = v48[0];
        v44 = v48[0];
        if ( v48[0] )
        {
          v28 = sub_6E9930(35, v48[0]);
          v29 = v28;
          v41 = v28;
        }
        else
        {
          v29 = v41;
        }
        sub_7016A0(v29, (_DWORD)v5, (unsigned int)v48, v44, a3, (unsigned int)&v46, v42);
        if ( *(_BYTE *)(a3 + 16) == 1 )
        {
          v30 = *(_QWORD *)(a3 + 144);
          if ( *(_BYTE *)(v30 + 24) )
            *(_BYTE *)(v30 + 59) |= 0x20u;
        }
        goto LABEL_40;
      }
      v9 = v48[0];
    }
    if ( !(unsigned int)sub_8D2D80(v9) && (!HIDWORD(qword_4F077B4) || !(unsigned int)sub_8D2B80(v48[0])) )
    {
      v15 = sub_6E94D0();
      sub_6E68E0(v15, v48);
LABEL_21:
      sub_6E6260(a3);
      goto LABEL_22;
    }
    if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 && (unsigned int)sub_6FCD00(v39, v5, v48, &v46, &v44, &v41) )
    {
      v45 = 0;
      v20 = v41;
      goto LABEL_39;
    }
    if ( HIDWORD(qword_4F077B4) && (unsigned int)sub_6FD310(v39, v5, v48, &v46, &v44, &v41) )
    {
      v45 = 0;
LABEL_38:
      v20 = v41;
LABEL_39:
      sub_7016A0(v20, (_DWORD)v5, (unsigned int)v48, v44, a3, (unsigned int)&v46, v42);
      goto LABEL_40;
    }
    v45 = sub_6E8B10(v5, v48, v12, v13, v14);
    v44 = v45;
    if ( !v45 )
      goto LABEL_38;
    v20 = sub_6E9930(v39, v45);
    v23 = v45;
    v41 = v20;
    if ( !v45 )
      goto LABEL_39;
    v24 = v20;
LABEL_55:
    sub_6FC7D0(v23, v5, v48, v24);
    v20 = v41;
    goto LABEL_39;
  }
  if ( (unsigned int)sub_8D2960(v48[0]) )
  {
    if ( (dword_4F077C0 || dword_4F077BC && qword_4F077A8 > 0x9DCFu)
      && ((v21 = sub_8D46C0(*v5), (unsigned int)sub_8D2600(v21))
       || (v26 = sub_8D46C0(*v5), (unsigned int)sub_8D2310(v26))) )
    {
      if ( *(char *)(qword_4D03C50 + 20LL) >= 0 )
        sub_69D070(0x477u, &v46);
    }
    else
    {
      sub_702E30(v5, 142);
    }
    v22 = *v5;
    v45 = v22;
    v44 = v22;
    if ( v22 )
    {
      v20 = sub_6E9930(v39, v22);
      v41 = v20;
      goto LABEL_39;
    }
    sub_7016A0(v41, (_DWORD)v5, (unsigned int)v48, 0, a3, (unsigned int)&v46, v42);
    goto LABEL_40;
  }
  if ( v39 != 36 || !(unsigned int)sub_8D2E30(v48[0]) )
  {
    v25 = sub_6E92F0();
    sub_6E68E0(v25, v48);
    goto LABEL_21;
  }
  v40 = sub_8D46C0(*v5);
  v31 = sub_8D46C0(v48[0]);
  if ( v40 == v31 || (unsigned int)sub_8DED30(v40, v31, 3) )
  {
    v32 = *v5;
    for ( j = *v5; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v45 = j;
    if ( dword_4F077C0 )
    {
      v37 = sub_8D46C0(v32);
      if ( (unsigned int)sub_8D2600(v37) || (v38 = sub_8D46C0(*v5), (unsigned int)sub_8D2310(v38)) )
      {
        if ( !(unsigned int)sub_6EA0D0(v5, v31)
          && !(unsigned int)sub_6EA0D0(v48, v31)
          && *(char *)(qword_4D03C50 + 20LL) >= 0 )
        {
          sub_69D070(0x477u, &v46);
        }
        goto LABEL_102;
      }
    }
    LODWORD(v6) = 0;
  }
  else if ( !(unsigned int)sub_6EB6C0(
                             (_DWORD)v5,
                             (unsigned int)v48,
                             (unsigned int)&v46,
                             byte_4B6D300[36],
                             0,
                             0,
                             0,
                             0,
                             (__int64)&v45) )
  {
LABEL_82:
    v44 = sub_72BA30(unk_4F06A60);
    goto LABEL_21;
  }
  if ( !(unsigned int)sub_702E30(v5, 142) || !(unsigned int)sub_702E30(v48, 142) )
    goto LABEL_82;
  if ( !v6 )
  {
LABEL_102:
    v44 = sub_72BA30(unk_4F06A60);
    v41 = 52;
    sub_7016A0(52, (_DWORD)v5, (unsigned int)v48, v44, a3, (unsigned int)&v46, v42);
    goto LABEL_40;
  }
  if ( !qword_4D0495C )
  {
    v35 = 5;
    if ( dword_4D04964 )
      v35 = byte_4F07472[0];
    sub_6E5D70(v35, 993, &v46, *v5, v48[0]);
  }
  v36 = sub_72BA30(unk_4F06A60);
  v23 = v45;
  v24 = 52;
  v41 = 52;
  v44 = v36;
  if ( v45 )
    goto LABEL_55;
  sub_7016A0(52, (_DWORD)v5, (unsigned int)v48, v36, a3, (unsigned int)&v46, v42);
LABEL_40:
  if ( (unsigned __int8)(v41 - 50) <= 1u )
    *(_QWORD *)(a3 + 88) = v5[11];
LABEL_22:
  v16 = *((_DWORD *)v5 + 17);
  *(_WORD *)(a3 + 72) = *((_WORD *)v5 + 36);
  *(_DWORD *)(a3 + 68) = v16;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a3 + 68);
  v17 = v50;
  *(_QWORD *)(a3 + 76) = v50;
  unk_4F061D8 = v17;
  return sub_6E3280(a3, &v46);
}
