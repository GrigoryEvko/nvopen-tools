// Function: sub_E99440
// Address: 0xe99440
//
__int64 __fastcall sub_E99440(__int64 a1)
{
  __int64 result; // rax

  result = sub_E99320(a1);
  if ( result )
    *(_BYTE *)(result + 80) = 1;
  return result;
}
