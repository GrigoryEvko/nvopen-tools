// Function: sub_C16510
// Address: 0xc16510
//
__int64 __fastcall sub_C16510(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // r8
  __int64 v3; // rsi

  v1 = *(_QWORD **)a1;
  v2 = 0;
  v3 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( v3 != *(_QWORD *)a1 )
  {
    do
    {
      switch ( *v1 )
      {
        case 1LL:
        case 6LL:
        case 7LL:
        case 8LL:
        case 9LL:
        case 0xBLL:
        case 0xCLL:
        case 0xDLL:
        case 0xELL:
        case 0xFLL:
        case 0x10LL:
        case 0x11LL:
        case 0x12LL:
        case 0x15LL:
        case 0x16LL:
        case 0x18LL:
        case 0x19LL:
        case 0x1ALL:
          v2 += 4;
          break;
        case 2LL:
        case 3LL:
        case 4LL:
        case 5LL:
        case 0xALL:
        case 0x13LL:
        case 0x14LL:
        case 0x17LL:
        case 0x1BLL:
          v2 += 8;
          break;
        default:
          BUG();
      }
      ++v1;
    }
    while ( (_QWORD *)v3 != v1 );
  }
  return v2;
}
