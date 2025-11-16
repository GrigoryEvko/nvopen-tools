// Function: sub_22418A0
// Address: 0x22418a0
//
__int64 __fastcall sub_22418A0(_QWORD *a1, char a2, unsigned __int64 a3)
{
  __int64 v4; // rdx
  __int64 result; // rax

  v4 = a1[1];
  result = -1;
  if ( v4 )
  {
    result = v4 - 1;
    if ( v4 - 1 > a3 )
      result = a3;
    do
    {
      if ( *(_BYTE *)(*a1 + result) == a2 )
        break;
    }
    while ( result-- != 0 );
  }
  return result;
}
