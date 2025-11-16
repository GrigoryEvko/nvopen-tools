// Function: sub_72F220
// Address: 0x72f220
//
__int64 *__fastcall sub_72F220(__int64 **a1)
{
  __int64 *result; // rax

  result = *a1;
  do
    result = (__int64 *)*result;
  while ( result && *((_BYTE *)result + 8) == 3 );
  *a1 = result;
  return result;
}
