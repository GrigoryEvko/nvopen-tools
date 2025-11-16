// Function: sub_38DD1F0
// Address: 0x38dd1f0
//
__int64 __fastcall sub_38DD1F0(__int64 a1)
{
  __int64 result; // rax

  result = sub_38DD140(a1);
  if ( result )
    *(_BYTE *)(result + 72) = 1;
  return result;
}
