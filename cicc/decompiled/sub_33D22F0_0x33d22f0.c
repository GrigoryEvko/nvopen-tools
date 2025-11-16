// Function: sub_33D22F0
// Address: 0x33d22f0
//
__int64 __fastcall sub_33D22F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v7; // edx

  result = sub_33D2050(a1, a2, a3, a4, a5, a6);
  if ( result )
  {
    v7 = *(_DWORD *)(result + 24);
    if ( v7 != 11 && v7 != 35 )
      return 0;
  }
  return result;
}
