// Function: sub_228DFC0
// Address: 0x228dfc0
//
char __fastcall sub_228DFC0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r15
  _BYTE *v5; // r12
  char result; // al
  _QWORD *v7; // rax
  __int16 v8; // ax
  __int64 v9; // rbx
  __int64 v10; // [rsp+10h] [rbp-40h]
  __int64 v11; // [rsp+18h] [rbp-38h]

  v4 = (_BYTE *)a4;
  v5 = (_BYTE *)a3;
  if ( a2 - 32 <= 1 )
  {
    v8 = *(_WORD *)(a3 + 24);
    if ( v8 == 4 )
    {
      if ( *(_WORD *)(a4 + 24) != 4 )
        goto LABEL_2;
    }
    else if ( v8 != 3 || *(_WORD *)(a4 + 24) != 3 )
    {
      goto LABEL_2;
    }
    v9 = *(_QWORD *)(a3 + 32);
    v11 = *(_QWORD *)(a4 + 32);
    v10 = sub_D95540(v9);
    if ( v10 == sub_D95540(v11) )
    {
      v4 = (_BYTE *)v11;
      v5 = (_BYTE *)v9;
    }
  }
LABEL_2:
  result = sub_DC3A60(*(_QWORD *)(a1 + 8), a2, v5, v4);
  if ( !result )
  {
    v7 = sub_DCC810(*(__int64 **)(a1 + 8), (__int64)v5, (__int64)v4, 0, 0);
    switch ( a2 )
    {
      case ' ':
        result = sub_D968A0((__int64)v7);
        break;
      case '!':
        result = sub_DBE090(*(_QWORD *)(a1 + 8), (__int64)v7);
        break;
      case '&':
        result = sub_DBEDC0(*(_QWORD *)(a1 + 8), (__int64)v7);
        break;
      case '\'':
        result = sub_DBED40(*(_QWORD *)(a1 + 8), (__int64)v7);
        break;
      case '(':
        result = sub_DBEC00(*(_QWORD *)(a1 + 8), (__int64)v7);
        break;
      case ')':
        result = sub_DBEC80(*(_QWORD *)(a1 + 8), (__int64)v7);
        break;
      default:
        BUG();
    }
  }
  return result;
}
