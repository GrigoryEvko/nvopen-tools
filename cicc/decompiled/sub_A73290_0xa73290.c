// Function: sub_A73290
// Address: 0xa73290
//
__int64 __fastcall sub_A73290(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( *a1 )
    result += 8LL * *(unsigned int *)(result + 8) + 64;
  return result;
}
