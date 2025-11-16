// Function: sub_6DEE30
// Address: 0x6dee30
//
__int64 __fastcall sub_6DEE30(__int64 a1)
{
  char v1; // al
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 1 )
  {
    result = *(_QWORD *)(a1 + 144);
    if ( !result )
      return 0;
    if ( *(_BYTE *)(result + 24) == 1 )
    {
LABEL_14:
      if ( *(_BYTE *)(result + 56) != 4 || (result = *(_QWORD *)(result + 72), *(_BYTE *)(result + 24) == 1) )
      {
        if ( (*(_BYTE *)(result + 27) & 2) != 0 && (*(_BYTE *)(result + 59) & 8) == 0 )
          return 0;
      }
    }
  }
  else
  {
    if ( v1 != 2 )
      return 0;
    result = *(_QWORD *)(a1 + 288);
    if ( result )
    {
      if ( *(_BYTE *)(result + 24) != 1 )
        return result;
      goto LABEL_14;
    }
    if ( *(_BYTE *)(a1 + 317) != 12 || *(_BYTE *)(a1 + 320) != 1 )
      return 0;
    result = sub_72E9A0(a1 + 144);
    if ( !result )
      return 0;
    if ( *(_BYTE *)(result + 24) == 1 )
    {
      if ( *(_BYTE *)(result + 56) != 4 || (result = *(_QWORD *)(result + 72), *(_BYTE *)(result + 24) == 1) )
      {
        if ( (*(_BYTE *)(result + 27) & 2) != 0 && (*(_BYTE *)(result + 59) & 8) == 0 )
          return 0;
      }
    }
  }
  return result;
}
