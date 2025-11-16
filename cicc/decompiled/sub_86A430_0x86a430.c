// Function: sub_86A430
// Address: 0x86a430
//
__int64 __fastcall sub_86A430(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // r8
  __int64 result; // rax
  __int64 v4; // rax
  char v5; // dl

  if ( !a1 )
    return 0;
  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 > 0xBu )
  {
    if ( v1 == 53 )
    {
      v4 = *(_QWORD *)(a1 + 24);
      v5 = *(_BYTE *)(v4 + 16);
      if ( ((v5 - 7) & 0xFB) == 0 )
        return *(_QWORD *)(v4 + 32);
      v2 = 0;
      if ( v5 == 6 )
      {
        v2 = *(_QWORD *)(v4 + 24);
        if ( *(_BYTE *)(v2 + 140) == 12 )
          return *(_QWORD *)(v2 + 160);
      }
      return v2;
    }
    return 0;
  }
  v2 = 0;
  if ( v1 <= 1u )
    return v2;
  switch ( v1 )
  {
    case 2u:
      result = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 128LL);
      break;
    case 6u:
      v2 = *(_QWORD *)(a1 + 24);
      if ( *(_BYTE *)(v2 + 140) == 12 )
        return *(_QWORD *)(v2 + 160);
      return v2;
    case 7u:
    case 8u:
      result = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 120LL);
      break;
    case 0xBu:
      result = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 152LL);
      break;
    default:
      return 0;
  }
  return result;
}
