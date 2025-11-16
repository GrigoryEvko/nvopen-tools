// Function: sub_131C740
// Address: 0x131c740
//
_DWORD *__fastcall sub_131C740(_DWORD *a1)
{
  _DWORD *result; // rax

  result = a1 + 36;
  do
    *a1++ = 1;
  while ( a1 != result );
  return result;
}
