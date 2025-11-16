// Function: sub_B159E0
// Address: 0xb159e0
//
__int64 __fastcall sub_B159E0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 v3; // dl
  __int64 result; // rax

  v2 = *a1;
  v3 = *(_BYTE *)(*a1 - 16);
  if ( (v3 & 2) != 0 )
  {
    result = **(_QWORD **)(v2 - 32);
    if ( !result )
      return result;
    return sub_B91420(result, a2);
  }
  result = *(_QWORD *)(v2 - 16 - 8LL * ((v3 >> 2) & 0xF));
  if ( result )
    return sub_B91420(result, a2);
  return result;
}
