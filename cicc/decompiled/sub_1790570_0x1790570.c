// Function: sub_1790570
// Address: 0x1790570
//
__int64 __fastcall sub_1790570(__int64 a1)
{
  __int64 v1; // rax
  char v2; // dl
  __int64 v3; // rax
  char v4; // dl
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 16) == 35 )
  {
    v1 = *(_QWORD *)(a1 - 48);
    v2 = *(_BYTE *)(v1 + 16);
    if ( v2 == 39 )
    {
      if ( *(_QWORD *)(v1 - 48) && *(_QWORD *)(v1 - 24) )
        return 0;
    }
    else if ( v2 == 5
           && *(_WORD *)(v1 + 18) == 15
           && *(_QWORD *)(v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF))
           && *(_QWORD *)(v1 + 24 * (1LL - (*(_DWORD *)(v1 + 20) & 0xFFFFFFF))) )
    {
      return 0;
    }
    v3 = *(_QWORD *)(a1 - 24);
    v4 = *(_BYTE *)(v3 + 16);
    if ( v4 == 39 )
    {
      if ( !*(_QWORD *)(v3 - 48) || !*(_QWORD *)(v3 - 24) )
        return 3;
    }
    else if ( v4 != 5
           || *(_WORD *)(v3 + 18) != 15
           || !*(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF))
           || !*(_QWORD *)(v3 + 24 * (1LL - (*(_DWORD *)(v3 + 20) & 0xFFFFFFF))) )
    {
      return 3;
    }
    return 0;
  }
  switch ( *(_BYTE *)(a1 + 16) )
  {
    case '%':
    case '/':
    case '0':
    case '1':
      result = 1;
      break;
    case '\'':
    case '2':
    case '3':
    case '4':
      return 3;
    default:
      return 0;
  }
  return result;
}
