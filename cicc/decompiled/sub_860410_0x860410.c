// Function: sub_860410
// Address: 0x860410
//
_BOOL8 __fastcall sub_860410(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  _BOOL8 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rdi

  while ( 1 )
  {
    if ( (*(_BYTE *)(a1 + 206) & 0x18) != 0 )
      return 1;
    if ( (*(_DWORD *)(a1 + 192) & 0x3001000) == (_DWORD)&loc_1000000 )
      break;
    v1 = *(_QWORD *)(a1 + 256);
    if ( !v1 || (v2 = *(_QWORD *)(v1 + 8)) == 0 )
    {
      result = 1;
      if ( (*(_BYTE *)(a1 + 193) & 0x20) == 0 && !*(_DWORD *)(a1 + 160) )
        return *(_QWORD *)(a1 + 344) != 0;
      return result;
    }
    a1 = v2;
  }
  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 32LL);
  switch ( *(_BYTE *)(v4 + 80) )
  {
    case 4:
    case 5:
      v5 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 80LL);
      break;
    case 6:
      v5 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v5 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v5 = *(_QWORD *)(v4 + 88);
      break;
    default:
      v5 = 0;
      break;
  }
  return *(_QWORD *)(sub_892400(v5) + 8) != 0;
}
