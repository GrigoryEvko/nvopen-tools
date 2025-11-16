// Function: sub_29D7D50
// Address: 0x29d7d50
//
__int64 __fastcall sub_29D7D50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v5; // r8d

  result = sub_29D7CF0(a1, *(unsigned int *)(a2 + 8), *(unsigned int *)(a3 + 8));
  if ( !(_DWORD)result )
  {
    v5 = sub_C49970(a2, (unsigned __int64 *)a3);
    result = 1;
    if ( v5 <= 0 )
      return (unsigned int)-((int)sub_C49970(a3, (unsigned __int64 *)a2) > 0);
  }
  return result;
}
