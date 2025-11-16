// Function: sub_1F3DAC0
// Address: 0x1f3dac0
//
__int64 __fastcall sub_1F3DAC0(__int64 a1, char a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  const char *v7; // rsi
  char v8; // al
  __int64 v9; // rdx
  _QWORD v11[6]; // [rsp+0h] [rbp-30h] BYREF

  v11[0] = a3;
  v11[1] = a4;
  if ( (_BYTE)a3 )
  {
    if ( (unsigned __int8)(a3 - 14) > 0x5Fu )
    {
LABEL_3:
      v5 = a1 + 16;
      v6 = 0;
      *(_QWORD *)a1 = a1 + 16;
      goto LABEL_4;
    }
  }
  else if ( !(unsigned __int8)sub_1F58D20(v11) )
  {
    goto LABEL_3;
  }
  v5 = a1 + 16;
  v6 = 4;
  *(_DWORD *)(a1 + 16) = 761488758;
  *(_QWORD *)a1 = a1 + 16;
LABEL_4:
  *(_QWORD *)(a1 + 8) = v6;
  v7 = "sqrt";
  *(_BYTE *)(v5 + v6) = 0;
  if ( !a2 )
    v7 = "div";
  if ( 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) < 4 - (unsigned __int64)(a2 == 0) )
    goto LABEL_21;
  sub_2241490(a1, v7);
  v8 = v11[0];
  if ( LOBYTE(v11[0]) )
  {
    if ( (unsigned __int8)(LOBYTE(v11[0]) - 14) <= 0x5Fu )
    {
      switch ( LOBYTE(v11[0]) )
      {
        case '^':
        case '_':
        case '`':
        case 'a':
        case 'j':
        case 'k':
        case 'l':
        case 'm':
          v9 = *(_QWORD *)(a1 + 8);
          goto LABEL_10;
        default:
          goto LABEL_15;
      }
    }
  }
  else
  {
    if ( !(unsigned __int8)sub_1F58D20(v11) )
    {
LABEL_15:
      v9 = *(_QWORD *)(a1 + 8);
      goto LABEL_16;
    }
    v8 = sub_1F596B0(v11);
  }
  v9 = *(_QWORD *)(a1 + 8);
  if ( v8 == 10 )
  {
LABEL_10:
    if ( v9 != 0x3FFFFFFFFFFFFFFFLL )
    {
      sub_2241490(a1, "d", 1);
      return a1;
    }
LABEL_21:
    sub_4262D8((__int64)"basic_string::append");
  }
LABEL_16:
  if ( v9 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_21;
  sub_2241490(a1, "f", 1);
  return a1;
}
