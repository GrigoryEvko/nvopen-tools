// Function: sub_15601B0
// Address: 0x15601b0
//
__int64 __fastcall sub_15601B0(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( *a1 )
    result += 8LL * *(unsigned int *)(result + 24) + 32;
  return result;
}
