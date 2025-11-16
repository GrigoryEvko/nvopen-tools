// Function: sub_8D57E0
// Address: 0x8d57e0
//
__int64 __fastcall sub_8D57E0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  *a2 = 0;
  result = *(_QWORD *)(a1 + 120);
  if ( !result )
    return 0;
  while ( 1 )
  {
    if ( *(_BYTE *)(result + 16) == 4 )
    {
      v3 = *(_QWORD *)(result + 8);
      if ( *(_BYTE *)(v3 + 140) == 12 )
      {
        if ( *(_QWORD *)(v3 + 8) )
          break;
      }
    }
    result = *(_QWORD *)result;
    if ( !result )
      return result;
  }
  *a2 = v3;
  return 1;
}
