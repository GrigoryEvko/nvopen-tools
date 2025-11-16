// Function: sub_69D0B0
// Address: 0x69d0b0
//
__int64 __fastcall sub_69D0B0(__int64 a1, __int64 a2, __int64 i, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  unsigned __int16 v8; // ax
  __int64 *v9; // r14
  int v10; // r13d
  int v11; // edx
  __int16 v12; // ax
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned __int8 v18; // dl
  __int64 j; // r8
  const char *v20; // rcx
  __int64 v21; // rax
  unsigned __int8 v22; // al
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  int v27; // [rsp+20h] [rbp-310h]
  int v28; // [rsp+24h] [rbp-30Ch]
  bool v29; // [rsp+28h] [rbp-308h]
  unsigned __int16 v30; // [rsp+2Ch] [rbp-304h]
  __int16 v31; // [rsp+2Eh] [rbp-302h]
  unsigned int v32; // [rsp+30h] [rbp-300h] BYREF
  unsigned int v33; // [rsp+34h] [rbp-2FCh] BYREF
  __int64 v34; // [rsp+38h] [rbp-2F8h] BYREF
  _BYTE v35[352]; // [rsp+40h] [rbp-2F0h] BYREF
  _BYTE v36[76]; // [rsp+1A0h] [rbp-190h] BYREF
  int v37; // [rsp+1ECh] [rbp-144h]
  __int16 v38; // [rsp+1F0h] [rbp-140h]

  v6 = i;
  v7 = a2;
  v32 = 0;
  if ( a2 )
  {
    v8 = *(_WORD *)(a2 + 8);
    a1 = a2;
    v9 = (__int64 *)v35;
    a2 = (__int64)v35;
    v30 = v8;
    sub_6F8AB0(a1, (unsigned int)v35, 0, 0, (unsigned int)&v34, (unsigned int)&v33, 0);
    v10 = *(_DWORD *)(*(_QWORD *)v7 + 44LL);
    v31 = *(_WORD *)(*(_QWORD *)v7 + 48LL);
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
      goto LABEL_3;
LABEL_18:
    if ( (unsigned int)sub_6E5430(a1, a2, i, a4, a5, a6) )
    {
      a2 = (__int64)&v34;
      sub_6851C0(0x39u, &v34);
    }
    sub_6E6260(v6);
    a1 = (__int64)v9;
    sub_6E6450(v9);
    goto LABEL_5;
  }
  v9 = (__int64 *)a1;
  v30 = word_4F06418[0];
  v34 = *(_QWORD *)&dword_4F063F8;
  v33 = dword_4F06650[0];
  v10 = qword_4F063F0;
  v31 = WORD2(qword_4F063F0);
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
    goto LABEL_18;
LABEL_3:
  if ( dword_4F077C4 != 2 )
    goto LABEL_4;
  a2 = 1;
  a1 = (__int64)v9;
  if ( !(unsigned int)sub_68FE10(v9, 1, 1) )
    goto LABEL_8;
  v27 = unk_4D04950;
  v29 = unk_4D04950 != 0;
  v28 = sub_8D2870(*v9);
  sub_6E7080(v36, 0);
  v37 = v10;
  a1 = byte_4B6D300[v30];
  v38 = v31;
  a2 = 0;
  sub_84EC30(
    a1,
    0,
    0,
    v27 == 0,
    (v28 | v29) != 0,
    (_DWORD)v9,
    (__int64)v36,
    (__int64)&v34,
    v33,
    0,
    0,
    v6,
    0,
    0,
    (__int64)&v32);
  v14 = v32;
  if ( v32 )
    goto LABEL_5;
  a1 = (unsigned int)a1;
  if ( v29 )
  {
    a2 = 1;
    sub_84EC30(a1, 1, 0, 0, 1, (_DWORD)v9, 0, (__int64)&v34, v33, 0, 0, v6, 0, 0, (__int64)&v32);
    a1 = (unsigned int)a1;
    if ( v32 )
    {
      if ( !*(_BYTE *)(v6 + 16) )
        goto LABEL_5;
      v21 = *(_QWORD *)v6;
      for ( i = *(unsigned __int8 *)(*(_QWORD *)v6 + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v21 + 140) )
        v21 = *(_QWORD *)(v21 + 160);
      if ( !(_BYTE)i )
        goto LABEL_5;
      v22 = 5;
      if ( unk_4D04950 )
        v22 = unk_4F07470;
      a1 = v22;
      a2 = 395;
      sub_6E5DE0(v22, 395, &v34, *(&off_4B6DFA0 + v30));
LABEL_8:
      v14 = v32;
      goto LABEL_9;
    }
    a2 = 0;
    sub_84EC30(a1, 0, 0, 1, v28, (_DWORD)v9, (__int64)v36, (__int64)&v34, v33, 0, 0, v6, 0, 0, (__int64)&v32);
    v14 = v32;
  }
LABEL_9:
  if ( v14 )
    goto LABEL_5;
  if ( dword_4F077C4 == 2 )
  {
    a1 = *v9;
    if ( (*(_BYTE *)(*v9 + 140) & 0xFB) == 8 )
    {
      a2 = 0;
      if ( (sub_8D4C10(a1, 0) & 2) != 0 )
      {
        a1 = 4;
        if ( dword_4F077C4 == 2 )
          a1 = (unsigned int)(unk_4F07778 > 202001) + 4;
        v20 = "an increment";
        a2 = 3010;
        if ( v30 != 31 )
          v20 = "a decrement";
        sub_6E5DE0(a1, 3010, (char *)v9 + 68, v20);
      }
LABEL_4:
      if ( v32 )
        goto LABEL_5;
    }
  }
  if ( (unsigned int)sub_6E9250(&v34) )
    goto LABEL_15;
  a1 = v32;
  if ( v32 )
    goto LABEL_5;
  a2 = 4;
  sub_6F69D0(v9, 4);
  if ( !(unsigned int)sub_6E96B0(v9) )
    goto LABEL_15;
  if ( (unsigned int)sub_8D2660(*v9) )
  {
    a2 = (__int64)v9;
    sub_6E68E0(2138, v9);
    goto LABEL_15;
  }
  if ( (unsigned int)sub_8D2E30(*v9) )
  {
    a2 = dword_4F077C0;
    if ( !dword_4F077C0
      || (v15 = sub_8D46C0(*v9), !(unsigned int)sub_8D2600(v15))
      && (v23 = sub_8D46C0(*v9), !(unsigned int)sub_8D2310(v23)) )
    {
      a2 = 142;
      if ( (unsigned int)sub_702E30(v9, 142) )
        goto LABEL_26;
LABEL_15:
      a1 = v6;
      sub_6E6260(v6);
      if ( v7 )
        goto LABEL_6;
LABEL_16:
      sub_7B8B50(a1, a2, i, a4);
      goto LABEL_6;
    }
    if ( *(char *)(qword_4D03C50 + 20LL) >= 0 )
    {
      a2 = 1143;
      if ( (unsigned int)sub_6E53E0(5, 1143, &v34) )
      {
        a2 = (__int64)&v34;
        sub_684B30(0x477u, &v34);
      }
    }
  }
  else
  {
    if ( dword_4F077C4 != 2 )
      goto LABEL_75;
    if ( (unsigned int)sub_8D2870(*v9) )
    {
      if ( unk_4D04950 )
      {
        a2 = 428;
        sub_6E5C80(unk_4F07470, 428, (char *)v9 + 68);
        goto LABEL_26;
      }
      a2 = (__int64)&v34;
      sub_69A8C0(511, &v34, v24, 0, v25, v26);
      goto LABEL_15;
    }
    if ( dword_4F077C4 == 2 && (unsigned int)sub_8D29A0(*v9) )
    {
      if ( v30 == 31 )
      {
        if ( unk_4D0476C )
        {
          a2 = (__int64)v9 + 68;
          sub_69D070(0x2C4u, (_DWORD *)v9 + 17);
        }
        else
        {
          a2 = (__int64)v9;
          sub_6E68E0(2800, v9);
        }
      }
      else
      {
        a2 = (__int64)v9;
        sub_6E68E0(709, v9);
      }
    }
    else
    {
LABEL_75:
      if ( (unsigned int)sub_8D2AF0(*v9) && (!HIDWORD(qword_4F077B4) || !(unsigned int)sub_8D2B50(*v9)) )
      {
        a2 = (__int64)v9;
        sub_6E68E0(1044, v9);
      }
    }
  }
LABEL_26:
  if ( !(unsigned int)sub_702F90(v9) )
    goto LABEL_15;
  sub_6ECF90(v9, 1);
  v16 = *v9;
  v17 = sub_73D720(*v9);
  v18 = *(_BYTE *)(v17 + 140);
  for ( j = v17; v18 == 12; v18 = *(_BYTE *)(v17 + 140) )
    v17 = *(_QWORD *)(v17 + 160);
  LOBYTE(a2) = (v30 != 31) + 35;
  if ( v18 > 3u )
  {
    if ( (unsigned __int8)(v18 - 5) > 1u )
LABEL_31:
      sub_721090(v16);
  }
  else if ( v18 <= 1u )
  {
    goto LABEL_31;
  }
  a2 = (unsigned __int8)a2;
  a1 = (__int64)v9;
  sub_6F7B30(v9, (unsigned __int8)a2, j, v6);
LABEL_5:
  if ( !v7 )
    goto LABEL_16;
LABEL_6:
  v11 = *((_DWORD *)v9 + 17);
  v12 = *((_WORD *)v9 + 36);
  *(_DWORD *)(v6 + 76) = v10;
  *(_WORD *)(v6 + 72) = v12;
  *(_DWORD *)(v6 + 68) = v11;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v6 + 68);
  *(_WORD *)(v6 + 80) = v31;
  unk_4F061D8 = *(_QWORD *)(v6 + 76);
  sub_6E3280(v6, &v34);
  sub_6E3BA0(v6, &v34, v33, 0);
  return sub_6E26D0(2, v6);
}
