// Function: sub_890400
// Address: 0x890400
//
__int64 __fastcall sub_890400(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rdx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 32);
  switch ( *(_BYTE *)(v1 + 80) )
  {
    case 4:
    case 5:
      v2 = *(_QWORD *)(*(_QWORD *)(v1 + 96) + 80LL);
      break;
    case 6:
      v2 = *(_QWORD *)(*(_QWORD *)(v1 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v2 = *(_QWORD *)(*(_QWORD *)(v1 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v2 = *(_QWORD *)(v1 + 88);
      break;
    default:
      v2 = 0;
      break;
  }
  result = (unsigned int)dword_4D04278;
  if ( dword_4D04278 )
  {
    result = 0;
    if ( (*(_BYTE *)(*(_QWORD *)(v2 + 104) + 121LL) & 2) == 0 )
      return *(_QWORD *)(a1 + 72) != 0;
  }
  return result;
}
