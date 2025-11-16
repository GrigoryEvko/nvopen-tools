// Function: sub_6B2B40
// Address: 0x6b2b40
//
__int64 __fastcall sub_6B2B40(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int16 v9; // ax
  __int64 v10; // rax
  __int64 v12; // rax
  char v13; // dl
  __int64 v14; // rax
  unsigned __int16 v15; // [rsp+Eh] [rbp-312h]
  unsigned __int8 v16; // [rsp+13h] [rbp-30Dh] BYREF
  unsigned int v17; // [rsp+14h] [rbp-30Ch] BYREF
  unsigned int v18; // [rsp+18h] [rbp-308h] BYREF
  int v19; // [rsp+1Ch] [rbp-304h] BYREF
  __int64 v20; // [rsp+20h] [rbp-300h] BYREF
  __int64 v21; // [rsp+28h] [rbp-2F8h] BYREF
  _BYTE v22[352]; // [rsp+30h] [rbp-2F0h] BYREF
  _QWORD v23[2]; // [rsp+190h] [rbp-190h] BYREF
  char v24; // [rsp+1A0h] [rbp-180h]
  int v25; // [rsp+1D4h] [rbp-14Ch] BYREF
  __int64 v26; // [rsp+1DCh] [rbp-144h]
  _BYTE v27[256]; // [rsp+220h] [rbp-100h] BYREF

  v19 = 0;
  if ( a2 )
  {
    v5 = (__int64 *)v22;
    v15 = *(_WORD *)(a2 + 8);
    sub_6F8AB0(a2, (unsigned int)v22, (unsigned int)v23, 0, (unsigned int)&v21, (unsigned int)&v17, 0);
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
  }
  else
  {
    v5 = a1;
    v15 = word_4F06418[0];
    v21 = *(_QWORD *)&dword_4F063F8;
    v17 = dword_4F06650[0];
    sub_7B8B50(a1, 0, a3, a4);
    sub_69ED20((__int64)v23, 0, 13, 0);
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
  }
  if ( (unsigned int)sub_68FE10(v5, 1, 1) || (unsigned int)sub_68FE10(v23, 0, 1) )
    sub_84EC30(
      byte_4B6D300[v15],
      0,
      0,
      1,
      0,
      (_DWORD)v5,
      (__int64)v23,
      (__int64)&v21,
      v17,
      0,
      0,
      a3,
      0,
      0,
      (__int64)&v19);
LABEL_3:
  if ( !v19 )
  {
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) != 2 || (sub_68BB70(v5, v23, &v21, a3, &v19), !v19) )
    {
      sub_6F69D0(v5, 0);
      sub_6F69D0(v23, 0);
      if ( !HIDWORD(qword_4F077B4) || !(unsigned int)sub_6FD310(v15, v5, v23, &v21, &v20, &v16) )
      {
        sub_6E93E0(v5);
        sub_6E9350(v23);
      }
      if ( dword_4F077C4 == 1 )
      {
        v20 = sub_6E8B10(v5, v23, v6, v7, v8);
        v16 = sub_6E9930(v15, v20);
        sub_6FC7D0(v20, v5, v23, v16);
        v12 = sub_72BA30(5);
        sub_6FC3F0(v12, v23, 1);
        if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) == 0 )
        {
LABEL_12:
          sub_7016A0(v16, (_DWORD)v5, (unsigned int)v23, v20, a3, (unsigned int)dword_4F07508, v17);
          goto LABEL_14;
        }
      }
      else
      {
        if ( !(unsigned int)sub_8D2B80(*v5) && !(unsigned int)sub_8D2B80(v23[0]) )
        {
          sub_6FC420(v5);
          sub_6FC420(v23);
          v20 = *v5;
          v16 = sub_6E9930(v15, v20);
        }
        if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) == 0 )
          goto LABEL_12;
      }
      if ( v24 == 2 && (v5[2] & 0xFD) != 0 )
      {
        v13 = *(_BYTE *)(*v5 + 140);
        if ( v13 == 12 )
        {
          v14 = *v5;
          do
          {
            v14 = *(_QWORD *)(v14 + 160);
            v13 = *(_BYTE *)(v14 + 140);
          }
          while ( v13 == 12 );
        }
        if ( v13 )
        {
          if ( v27[173] == 1 )
          {
            sub_7131E0(v27, *v5, &v18);
            if ( v18 )
              sub_69D070(v18, &v25);
          }
        }
      }
      goto LABEL_12;
    }
  }
LABEL_14:
  v9 = *((_WORD *)v5 + 36);
  *(_DWORD *)(a3 + 68) = *((_DWORD *)v5 + 17);
  *(_WORD *)(a3 + 72) = v9;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a3 + 68);
  v10 = v26;
  *(_QWORD *)(a3 + 76) = v26;
  unk_4F061D8 = v10;
  return sub_6E3280(a3, &v21);
}
