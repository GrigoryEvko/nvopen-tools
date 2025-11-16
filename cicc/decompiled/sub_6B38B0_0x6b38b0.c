// Function: sub_6B38B0
// Address: 0x6b38b0
//
__int64 __fastcall sub_6B38B0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r14d
  _BYTE *v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  int v10; // edx
  __int64 v11; // rax
  unsigned int v13; // r15d
  unsigned __int8 v14; // [rsp+7h] [rbp-309h] BYREF
  unsigned int v15; // [rsp+8h] [rbp-308h] BYREF
  int v16; // [rsp+Ch] [rbp-304h] BYREF
  __int64 v17; // [rsp+10h] [rbp-300h] BYREF
  __int64 v18; // [rsp+18h] [rbp-2F8h] BYREF
  _BYTE v19[352]; // [rsp+20h] [rbp-2F0h] BYREF
  _QWORD v20[9]; // [rsp+180h] [rbp-190h] BYREF
  __int64 v21; // [rsp+1CCh] [rbp-144h]

  v16 = 0;
  if ( a2 )
  {
    v5 = *(unsigned __int16 *)(a2 + 8);
    v6 = v19;
    sub_6F8AB0(a2, (unsigned int)v19, (unsigned int)v20, 0, (unsigned int)&v18, (unsigned int)&v15, 0);
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
    goto LABEL_13;
  }
  v6 = a1;
  v5 = word_4F06418[0];
  v18 = *(_QWORD *)&dword_4F063F8;
  v15 = dword_4F06650[0];
  switch ( word_4F06418[0] )
  {
    case '2':
      v13 = 7;
      break;
    case '3':
      v13 = 6;
      break;
    case '!':
      v13 = 8;
      break;
    default:
      sub_721090(a1);
  }
  sub_7B8B50(a1, 0, a3, a4);
  sub_69ED20((__int64)v20, 0, v13, 0);
  if ( dword_4F077C4 == 2 )
  {
LABEL_13:
    if ( (unsigned int)sub_68FE10(v6, 1, 1) || (unsigned int)sub_68FE10(v20, 0, 1) )
      sub_84EC30(
        byte_4B6D300[(unsigned __int16)v5],
        0,
        0,
        1,
        0,
        (_DWORD)v6,
        (__int64)v20,
        (__int64)&v18,
        v15,
        0,
        0,
        a3,
        0,
        0,
        (__int64)&v16);
  }
LABEL_3:
  if ( !v16 )
  {
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) != 2 || (sub_68BB70(v6, v20, &v18, a3, &v16), !v16) )
    {
      sub_6F69D0(v6, 0);
      sub_6F69D0(v20, 0);
      if ( !HIDWORD(qword_4F077B4) || !(unsigned int)sub_6FD310(v5, v6, v20, &v18, &v17, &v14) )
      {
        sub_6E9350(v6);
        sub_6E9350(v20);
        v17 = sub_6E8B10(v6, v20, v7, v8, v9);
        v14 = sub_6E9930(v5, v17);
        sub_6FC7D0(v17, v6, v20, v14);
      }
      sub_7016A0(v14, (_DWORD)v6, (unsigned int)v20, v17, a3, (unsigned int)&v18, v15);
    }
  }
  v10 = *((_DWORD *)v6 + 17);
  *(_WORD *)(a3 + 72) = *((_WORD *)v6 + 36);
  *(_DWORD *)(a3 + 68) = v10;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a3 + 68);
  v11 = v21;
  *(_QWORD *)(a3 + 76) = v21;
  unk_4F061D8 = v11;
  return sub_6E3280(a3, &v18);
}
