// Function: sub_A74460
// Address: 0xa74460
//
__int64 __fastcall sub_A74460(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( *a1 )
    result += 8LL * *(unsigned int *)(result + 8) + 48;
  return result;
}
