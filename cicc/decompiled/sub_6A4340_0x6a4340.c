// Function: sub_6A4340
// Address: 0x6a4340
//
__int64 __fastcall sub_6A4340(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  unsigned __int16 v7; // bx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // eax
  __int64 v17; // rdi
  const char *v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int8 v23; // dl
  __int64 i; // r8
  __int64 v25; // rax
  char j; // dl
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // r9
  unsigned int v33; // [rsp+Ch] [rbp-1A4h]
  unsigned int v34; // [rsp+10h] [rbp-1A0h] BYREF
  int v35; // [rsp+14h] [rbp-19Ch] BYREF
  __int64 v36; // [rsp+18h] [rbp-198h] BYREF
  _QWORD v37[8]; // [rsp+20h] [rbp-190h] BYREF
  int v38; // [rsp+64h] [rbp-14Ch] BYREF
  __int64 v39; // [rsp+6Ch] [rbp-144h]
  __int64 v40; // [rsp+78h] [rbp-138h]

  v6 = a2;
  v35 = 0;
  if ( !a1 )
  {
    v7 = word_4F06418[0];
    v36 = *(_QWORD *)&dword_4F063F8;
    v34 = dword_4F06650[0];
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
    {
      sub_7B8B50(0, a2, a3, a4);
      sub_69ED20((__int64)v37, 0, 18, 32);
      goto LABEL_3;
    }
    if ( (unsigned int)sub_6E5430(0, a2, a3, a4, a5, a6) )
    {
      a2 = &v36;
      a1 = 57;
      sub_6851C0(0x39u, &v36);
    }
    sub_7B8B50(a1, a2, v14, v15);
    sub_69ED20((__int64)v37, 0, 18, 32);
LABEL_15:
    sub_6E6260(v6);
    sub_6E6450(v37);
    goto LABEL_10;
  }
  v7 = *(_WORD *)(a1 + 8);
  sub_6F8AB0(a1, (unsigned int)v37, 0, 0, (unsigned int)&v36, (unsigned int)&v34, 0);
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_6E5430(a1, v37, v8, v9, v10, v11) )
      sub_6851C0(0x39u, &v36);
    goto LABEL_15;
  }
LABEL_3:
  if ( dword_4F077C4 == 2 && (unsigned int)sub_68FE10(v37, 1, 1) )
  {
    v33 = v34;
    v16 = sub_8D2870(v37[0]);
    sub_84EC30(
      byte_4B6D300[v7],
      1,
      0,
      1,
      v16,
      (unsigned int)v37,
      0,
      (__int64)&v36,
      v33,
      0,
      0,
      (__int64)a2,
      0,
      0,
      (__int64)&v35);
  }
  if ( v35 )
    goto LABEL_10;
  if ( dword_4F077C4 == 2 && (*(_BYTE *)(v37[0] + 140LL) & 0xFB) == 8 )
  {
    if ( (sub_8D4C10(v37[0], 0) & 2) != 0 )
    {
      v17 = 4;
      if ( dword_4F077C4 == 2 )
        v17 = (unsigned int)(unk_4F07778 > 202001) + 4;
      v18 = "an increment";
      if ( v7 != 31 )
        v18 = "a decrement";
      sub_6E5DE0(v17, 3010, &v38, v18);
    }
    if ( v35 )
      goto LABEL_10;
  }
  if ( (unsigned int)sub_6E9250(&v36) )
  {
LABEL_9:
    sub_6E6260(a2);
    goto LABEL_10;
  }
  if ( v35 )
    goto LABEL_10;
  sub_6F69D0(v37, 4);
  if ( !(unsigned int)sub_6E96B0(v37) )
    goto LABEL_9;
  if ( (unsigned int)sub_8D2660(v37[0]) )
  {
    sub_6E68E0(2138, v37);
    goto LABEL_9;
  }
  if ( (unsigned int)sub_8D2E30(v37[0]) )
  {
    if ( dword_4F077C0
      && ((v19 = sub_8D46C0(v37[0]), (unsigned int)sub_8D2600(v19))
       || (v28 = sub_8D46C0(v37[0]), (unsigned int)sub_8D2310(v28))) )
    {
      if ( *(char *)(qword_4D03C50 + 20LL) >= 0 && (unsigned int)sub_6E53E0(5, 1143, &v36) )
        sub_684B30(0x477u, &v36);
    }
    else if ( !(unsigned int)sub_702E30(v37, 142) )
    {
      goto LABEL_9;
    }
    goto LABEL_33;
  }
  v27 = v37[0];
  if ( dword_4F077C4 != 2 )
    goto LABEL_48;
  if ( (unsigned int)sub_8D2870(v37[0]) )
  {
    if ( !unk_4D04950 )
    {
      sub_69A8C0(511, &v36, v30, 0, v31, v32);
      goto LABEL_9;
    }
    sub_6E5C80(unk_4F07470, 428, &v38);
    goto LABEL_33;
  }
  v27 = v37[0];
  if ( dword_4F077C4 != 2 )
    goto LABEL_48;
  if ( !(unsigned int)sub_8D29A0(v37[0]) )
  {
    v27 = v37[0];
LABEL_48:
    if ( (unsigned int)sub_8D2AF0(v27) && (!HIDWORD(qword_4F077B4) || !(unsigned int)sub_8D2B50(v37[0])) )
      sub_6E68E0(1044, v37);
    goto LABEL_33;
  }
  if ( v7 == 31 )
  {
    if ( unk_4D0476C )
      sub_69D070(0x2C4u, &v38);
    else
      sub_6E68E0(2800, v37);
  }
  else
  {
    sub_6E68E0(709, v37);
  }
LABEL_33:
  if ( !(unsigned int)sub_702F90(v37) )
    goto LABEL_9;
  sub_6ECF90(v37, 1);
  v20 = v37[0];
  v21 = v37[0];
  v22 = sub_73D720(v37[0]);
  v23 = *(_BYTE *)(v22 + 140);
  for ( i = v22; v23 == 12; v23 = *(_BYTE *)(v22 + 140) )
    v22 = *(_QWORD *)(v22 + 160);
  if ( v23 > 3u )
  {
    if ( (unsigned __int8)(v23 - 5) > 1u )
LABEL_38:
      sub_721090(v21);
  }
  else if ( v23 <= 1u )
  {
    goto LABEL_38;
  }
  sub_6F7B30(v37, (unsigned __int8)((v7 != 31) + 37), i, a2);
  if ( dword_4F077C4 == 2 )
  {
    if ( !*((_BYTE *)a2 + 16) )
      goto LABEL_46;
    v25 = *a2;
    for ( j = *(_BYTE *)(*a2 + 140); j == 12; j = *(_BYTE *)(v25 + 140) )
      v25 = *(_QWORD *)(v25 + 160);
    if ( j )
    {
      v29 = a2[18];
      *(_BYTE *)(v29 + 25) |= 1u;
      *(_BYTE *)(v29 + 58) |= 1u;
      *a2 = v20;
      *(_QWORD *)v29 = v20;
      a2[11] = v40;
      sub_6E6A20(a2);
    }
    else
    {
LABEL_46:
      sub_6E6870(a2);
    }
  }
LABEL_10:
  *((_DWORD *)v6 + 17) = v36;
  *((_WORD *)v6 + 36) = WORD2(v36);
  *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v6 + 68);
  v12 = v39;
  *(__int64 *)((char *)v6 + 76) = v39;
  unk_4F061D8 = v12;
  sub_6E3280(v6, &v36);
  sub_6E3BA0(v6, &v36, v34, 0);
  return sub_6E26D0(2, v6);
}
