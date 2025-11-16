// Function: sub_8794A0
// Address: 0x8794a0
//
__int64 __fastcall sub_8794A0(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // rcx

  result = a1[21];
  if ( !result )
  {
    v2 = *a1;
    switch ( *(_BYTE *)(*a1 + 80LL) )
    {
      case 4:
      case 5:
        result = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 80LL);
        break;
      case 6:
        result = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        result = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        result = *(_QWORD *)(v2 + 88);
        break;
      default:
        return result;
    }
  }
  return result;
}
