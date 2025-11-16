// Function: sub_B5B5E0
// Address: 0xb5b5e0
//
__int64 __fastcall sub_B5B5E0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 - 32);
  if ( v1 && !*(_BYTE *)v1 && *(_QWORD *)(v1 + 24) == *(_QWORD *)(a1 + 80) )
  {
    switch ( *(_DWORD *)(v1 + 36) )
    {
      case 0x137:
      case 0x138:
      case 0x167:
      case 0x168:
        return 13;
      case 0x14D:
      case 0x171:
        return 17;
      case 0x152:
      case 0x153:
      case 0x173:
      case 0x174:
        return 15;
      default:
        break;
    }
  }
  BUG();
}
