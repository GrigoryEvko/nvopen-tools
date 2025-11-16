// Function: sub_33E1670
// Address: 0x33e1670
//
__int64 __fastcall sub_33E1670(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v7; // edx

  result = sub_33D2050(a1, a2, a3, a4, a5, a6);
  if ( result )
  {
    v7 = *(_DWORD *)(result + 24);
    if ( v7 != 12 && v7 != 36 )
      return 0;
  }
  return result;
}
