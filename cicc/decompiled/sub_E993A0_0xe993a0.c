// Function: sub_E993A0
// Address: 0xe993a0
//
__int64 __fastcall sub_E993A0(__int64 a1)
{
  __int64 result; // rax

  result = sub_E99320(a1);
  if ( result )
    *(_BYTE *)(result + 88) = 1;
  return result;
}
