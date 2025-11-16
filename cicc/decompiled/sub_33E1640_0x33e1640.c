// Function: sub_33E1640
// Address: 0x33e1640
//
__int64 __fastcall sub_33E1640(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v7; // edx

  result = sub_33D2250(a1, a2, a3, a4, a5, a6);
  if ( result )
  {
    v7 = *(_DWORD *)(result + 24);
    if ( v7 != 11 && v7 != 35 )
      return 0;
  }
  return result;
}
