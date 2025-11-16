// Function: sub_8E5680
// Address: 0x8e5680
//
void __fastcall sub_8E5680(__int64 a1, char a2)
{
  if ( a2 == 21 )
  {
    switch ( *(_BYTE *)(a1 + 40) )
    {
      case 6:
        sub_6851A0(0xED2u, (_DWORD *)a1, (__int64)"goto statement");
        break;
      case 9:
        sub_6851A0(0xED2u, (_DWORD *)a1, (__int64)"coroutine statement");
        break;
      case 0xA:
        sub_6851A0(0xED2u, (_DWORD *)a1, (__int64)"co_return statement");
        break;
      case 0x12:
        sub_6851A0(0xED2u, (_DWORD *)a1, (__int64)"asm statement");
        break;
      case 0x13:
        sub_6851A0(0xED2u, (_DWORD *)a1, (__int64)"try block");
        break;
      case 0x15:
      case 0x16:
        sub_6851A0(0xED2u, (_DWORD *)a1, (__int64)"variable length array");
        break;
      default:
        return;
    }
  }
}
