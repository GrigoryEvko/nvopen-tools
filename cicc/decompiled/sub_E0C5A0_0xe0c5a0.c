// Function: sub_E0C5A0
// Address: 0xe0c5a0
//
__int64 __fastcall sub_E0C5A0(__int64 a1, __int64 a2)
{
  if ( a2 != 17 )
  {
    switch ( a2 )
    {
      case 16LL:
        if ( !(*(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6665646E755F4F46LL) )
          return 2;
        break;
      case 21LL:
        if ( !(*(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x74726174735F4F46LL)
          && *(_DWORD *)(a1 + 16) == 1818846815
          && *(_BYTE *)(a1 + 20) == 101 )
        {
          return 3;
        }
        if ( !(*(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6F646E65765F4F46LL)
          && *(_DWORD *)(a1 + 16) == 2019909490
          && *(_BYTE *)(a1 + 20) == 116 )
        {
          return 255;
        }
        break;
      case 19LL:
        if ( !(*(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x665F646E655F4F46LL)
          && *(_WORD *)(a1 + 16) == 27753
          && *(_BYTE *)(a1 + 18) == 101 )
        {
          return 4;
        }
        break;
      default:
        return 0xFFFFFFFFLL;
    }
    return 0xFFFFFFFFLL;
  }
  if ( *(_QWORD *)a1 ^ 0x4E4943414D5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6E696665645F4F46LL || *(_BYTE *)(a1 + 16) != 101 )
    return 0xFFFFFFFFLL;
  return 1;
}
