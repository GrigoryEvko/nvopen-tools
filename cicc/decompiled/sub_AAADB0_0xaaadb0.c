// Function: sub_AAADB0
// Address: 0xaaadb0
//
__int64 __fastcall sub_AAADB0(__int64 a1, unsigned int *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rbx

  while ( 1 )
  {
    result = a1;
    if ( !a3 )
      break;
    v4 = a3;
    result = sub_AD69F0(a1, *a2);
    if ( !result )
      break;
    if ( v4 == 1 )
      break;
    result = sub_AD69F0(result, a2[1]);
    if ( !result )
      break;
    if ( v4 == 2 )
      break;
    result = sub_AD69F0(result, a2[2]);
    if ( !result )
      break;
    a2 += 3;
    a3 = v4 - 3;
    a1 = result;
  }
  return result;
}
