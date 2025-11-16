// Function: sub_A73120
// Address: 0xa73120
//
__int64 __fastcall sub_A73120(__int64 *a1, __int64 *a2)
{
  char v2; // r8
  __int64 result; // rax

  v2 = sub_A730F0(a1, *a2);
  result = 0xFFFFFFFFLL;
  if ( !v2 )
    return (unsigned __int8)sub_A730F0(a2, *a1);
  return result;
}
