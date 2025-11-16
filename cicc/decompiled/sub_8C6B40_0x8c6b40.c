// Function: sub_8C6B40
// Address: 0x8c6b40
//
__int64 __fastcall sub_8C6B40(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rdx
  char v5; // cl
  char v6[9]; // [rsp+Fh] [rbp-11h] BYREF

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 0:
    case 1:
    case 0xC:
    case 0xD:
    case 0x10:
    case 0x18:
      return 0;
    case 2:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xB:
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x17:
      goto LABEL_2;
    case 3:
      v4 = *(_QWORD *)(a1 + 88);
      if ( *(_BYTE *)(v4 + 140) != 12 || !*(_QWORD *)(v4 + 8) )
        return 0;
      do
      {
        v4 = *(_QWORD *)(v4 + 160);
        v5 = *(_BYTE *)(v4 + 140);
      }
      while ( v5 == 12 );
      if ( (unsigned __int8)(v5 - 9) <= 2u )
        return (*(_BYTE *)(v4 + 177) & 4) != 0;
      result = 0;
      if ( v5 == 2 && (*(_BYTE *)(v4 + 161) & 8) != 0 )
        return (*(_BYTE *)(v4 + 162) & 8) != 0;
      return result;
    case 4:
    case 5:
    case 6:
      if ( dword_4F077C4 != 2 )
        return 1;
LABEL_2:
      v1 = sub_87D1A0(a1, v6);
      result = 0;
      if ( v1 )
        return (*(_BYTE *)(v1 + 88) & 0x60) == 32;
      return result;
    case 0xE:
    case 0xF:
      sub_8D2310(**(_QWORD **)(a1 + 88));
      return (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL) + 88LL) & 0x60) == 32;
    case 0x11:
      v3 = *(_QWORD *)(a1 + 88);
      if ( !v3 )
        return 0;
      break;
    default:
      sub_721090();
  }
  while ( !(unsigned int)sub_8C6B40(v3) )
  {
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      return 0;
  }
  return 1;
}
