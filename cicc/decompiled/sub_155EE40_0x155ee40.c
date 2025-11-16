// Function: sub_155EE40
// Address: 0x155ee40
//
__int64 __fastcall sub_155EE40(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( *a1 )
    result += 8LL * *(unsigned int *)(result + 16) + 24;
  return result;
}
