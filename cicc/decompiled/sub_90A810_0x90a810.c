// Function: sub_90A810
// Address: 0x90a810
//
__int64 __fastcall sub_90A810(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r8
  __int64 v5; // rdi

  v4 = a4;
  v5 = *a1;
  if ( !a4 )
  {
    v4 = 0;
    a3 = 0;
  }
  return sub_B6E160(v5, a2, a3, v4);
}
