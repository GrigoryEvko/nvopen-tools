// Function: sub_1D1AD70
// Address: 0x1d1ad70
//
__int64 __fastcall sub_1D1AD70(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 result; // rax
  int v7; // edx

  result = sub_1D1AA50(a1, a2, a3, a4, a5, a6);
  if ( result )
  {
    v7 = *(unsigned __int16 *)(result + 24);
    if ( v7 != 10 && v7 != 32 )
      return 0;
  }
  return result;
}
