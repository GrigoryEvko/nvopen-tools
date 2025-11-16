// Function: sub_69A120
// Address: 0x69a120
//
__int64 __fastcall sub_69A120(char a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v4; // r13d
  char v6; // al
  __int64 v7; // rsi
  __int64 v8; // rax
  char v9; // r14
  bool v10; // r14
  _BYTE *v11; // rdi
  unsigned int v12; // r12d
  __int64 v13; // [rsp+8h] [rbp-D8h] BYREF
  _BYTE v14[17]; // [rsp+10h] [rbp-D0h] BYREF
  char v15; // [rsp+21h] [rbp-BFh]

  v2 = a2;
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a2) )
    sub_8AE000(a2);
  if ( (unsigned int)sub_8D2310(a2) )
    return 0;
  v4 = sub_8D23E0(a2);
  if ( v4 )
    return 0;
  if ( (unsigned int)sub_8D3410(a2) )
    v2 = sub_8D40F0(a2);
  while ( 1 )
  {
    v6 = *(_BYTE *)(v2 + 140);
    if ( v6 != 12 )
      break;
    v2 = *(_QWORD *)(v2 + 160);
  }
  if ( (*(_BYTE *)(v2 + 141) & 0x20) != 0 )
  {
    if ( (!dword_4F077BC || (_DWORD)qword_4F077B4)
      && ((unsigned __int8)(v6 - 9) <= 2u || v6 == 2 && (*(_BYTE *)(v2 + 161) & 8) != 0) )
    {
      sub_6E5F60(dword_4F07508, v2, 8);
      return v4;
    }
    return 0;
  }
  v4 = 1;
  if ( (unsigned __int8)(v6 - 9) <= 2u )
  {
    sub_6E1DD0(&v13);
    sub_6E1E00(5, v14, 0, 1);
    v7 = v2;
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x80u;
    v15 &= 0xFCu;
    v8 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v9 = *(_BYTE *)(v8 + 7);
    *(_BYTE *)(v8 + 7) = v9 & 0xF7;
    v10 = (v9 & 8) != 0;
    v11 = (_BYTE *)sub_6EB2F0(v2, v2, &dword_4F063F8, 1);
    v12 = (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0;
    if ( v11 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 )
    {
      if ( (v11[206] & 0x10) != 0 || (v11[88] & 3) != 0 )
      {
        v12 = 0;
      }
      else if ( a1 == 43 )
      {
        v12 = sub_8D7760();
      }
      else if ( a1 == 44 )
      {
        v12 = (v11[194] & 8) != 0;
      }
    }
    v4 = v12;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7)
                                                             & 0xF7
                                                             | (8 * v10);
    sub_6E2B30(v11, v7);
    sub_6E1DF0(v13);
  }
  return v4;
}
