// Function: sub_8D1DB0
// Address: 0x8d1db0
//
__int64 __fastcall sub_8D1DB0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = sub_8D1CF0(a1, a2);
  if ( !(_DWORD)result && !*a2 )
    return sub_8D1B60(a1, a2);
  return result;
}
