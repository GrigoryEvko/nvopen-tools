// Function: sub_72AD30
// Address: 0x72ad30
//
__int64 __fastcall sub_72AD30(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl

  result = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL);
  if ( result )
  {
    while ( 1 )
    {
      v2 = *(_BYTE *)(result + 28);
      if ( v2 != 6 )
        break;
      result = *(_QWORD *)(result + 16);
      if ( !result )
        return result;
    }
    if ( v2 == 3 )
      return *(_QWORD *)(result + 32);
    else
      return 0;
  }
  return result;
}
