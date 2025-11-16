// Function: sub_DAEA10
// Address: 0xdaea10
//
__int64 __fastcall sub_DAEA10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r14
  __int64 v5; // rdx
  char v6; // r15
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  char v10; // al
  __int64 *v11; // [rsp-40h] [rbp-40h]

  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
    case 1:
      return 2;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xE:
      goto LABEL_3;
    case 8:
      if ( !(unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 40), **(_QWORD **)(*(_QWORD *)(a2 + 48) + 32LL), a3) )
        return 0;
LABEL_3:
      v4 = (__int64 *)sub_D960E0(a2);
      v11 = &v4[v5];
      if ( v4 == v11 )
        return 2;
      v6 = 1;
      break;
    case 0xF:
      v8 = *(_QWORD *)(a2 - 8);
      if ( *(_BYTE *)v8 <= 0x1Cu )
        return 2;
      v9 = *(_QWORD *)(v8 + 40);
      if ( a3 == v9 )
        return 1;
      sub_B196A0(*(_QWORD *)(a1 + 40), v9, a3);
      if ( v10 )
        return 2;
      return 0;
    default:
      BUG();
  }
  while ( 1 )
  {
    result = sub_DAE3E0(a1, *v4, a3);
    if ( !(_DWORD)result )
      break;
    if ( (_DWORD)result == 1 )
      v6 = 0;
    if ( v11 == ++v4 )
    {
      if ( v6 )
        return 2;
      else
        return 1;
    }
  }
  return result;
}
