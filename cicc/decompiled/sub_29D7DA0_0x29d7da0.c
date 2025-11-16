// Function: sub_29D7DA0
// Address: 0x29d7da0
//
__int64 __fastcall sub_29D7DA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = sub_29D7D50(a1, a2, a3);
  if ( !(_DWORD)result )
    return sub_29D7D50(a1, a2 + 16, a3 + 16);
  return result;
}
