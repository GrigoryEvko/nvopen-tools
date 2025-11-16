// Function: sub_1584BD0
// Address: 0x1584bd0
//
__int64 __fastcall sub_1584BD0(__int64 a1, unsigned int *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rbx

  while ( 1 )
  {
    result = a1;
    if ( !a3 )
      break;
    v4 = a3;
    result = sub_15A0A60(a1, *a2);
    if ( !result )
      break;
    if ( v4 == 1 )
      break;
    result = sub_15A0A60(result, a2[1]);
    if ( !result )
      break;
    if ( v4 == 2 )
      break;
    result = sub_15A0A60(result, a2[2]);
    if ( !result )
      break;
    a2 += 3;
    a3 = v4 - 3;
    a1 = result;
  }
  return result;
}
