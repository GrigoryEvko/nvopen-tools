// Function: sub_1EE5F10
// Address: 0x1ee5f10
//
__int64 __fastcall sub_1EE5F10(__int64 *a1)
{
  __int64 result; // rax

  a1[24] = 0;
  result = *a1;
  a1[23] = 0;
  if ( result != a1[1] )
    a1[1] = result;
  *((_DWORD *)a1 + 8) = 0;
  *((_DWORD *)a1 + 28) = 0;
  return result;
}
