// Function: sub_E993C0
// Address: 0xe993c0
//
__int64 __fastcall sub_E993C0(__int64 a1)
{
  __int64 result; // rax

  result = sub_E99320(a1);
  if ( result )
    *(_BYTE *)(result + 89) = 1;
  return result;
}
