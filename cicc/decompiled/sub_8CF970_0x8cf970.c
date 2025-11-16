// Function: sub_8CF970
// Address: 0x8cf970
//
__int64 __fastcall sub_8CF970(__int64 a1, unsigned __int8 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax

  if ( (*(_BYTE *)(a1 - 8) & 2) == 0 )
  {
    v3 = *(_QWORD *)(a1 + 32);
    if ( v3 )
      return *(_QWORD *)v3;
    else
      return a1;
  }
  if ( a2 > 0x1Cu )
  {
    if ( a2 == 59 )
      return sub_8C9880(a1);
LABEL_27:
    sub_721090();
  }
  if ( a2 <= 5u )
    goto LABEL_27;
  switch ( a2 )
  {
    case 6u:
      return sub_8CA330(a1);
    case 7u:
      return sub_8C97D0(a1);
    case 8u:
      if ( !*qword_4D03FD0 )
        return a1;
      if ( !unk_4D03FC4 && (!dword_4D03FC0 || (*(_BYTE *)(a1 + 89) & 4) == 0) )
        break;
      v4 = *(_QWORD *)(a1 + 32);
      if ( v4 )
        return *(_QWORD *)v4;
      sub_8C9400(a1, 8);
      break;
    case 0xBu:
      return sub_8CA280(a1);
    case 0x1Cu:
      if ( !*qword_4D03FD0 )
        return a1;
      if ( !unk_4D03FC4 && (!dword_4D03FC0 || (*(_BYTE *)(a1 + 89) & 4) == 0) )
        break;
      v4 = *(_QWORD *)(a1 + 32);
      if ( v4 )
        return *(_QWORD *)v4;
      sub_8C9400(a1, 28);
      break;
    default:
      goto LABEL_27;
  }
  v4 = *(_QWORD *)(a1 + 32);
  if ( v4 )
    return *(_QWORD *)v4;
  else
    return a1;
}
