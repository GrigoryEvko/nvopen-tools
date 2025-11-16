// Function: sub_B43CE0
// Address: 0xb43ce0
//
__int64 __fastcall sub_B43CE0(__int64 a1)
{
  __int64 result; // rax
  __int64 *v2; // rdi

  result = *(_QWORD *)(a1 + 40);
  if ( *(_BYTE *)(result + 40) )
  {
    v2 = *(__int64 **)(a1 + 64);
    if ( v2 )
      return sub_B144A0(v2);
  }
  return result;
}
