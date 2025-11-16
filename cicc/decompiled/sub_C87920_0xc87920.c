// Function: sub_C87920
// Address: 0xc87920
//
__int64 __fastcall sub_C87920(char a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdi

  v5 = a1 & 1;
  if ( !(_DWORD)v5 )
    return sub_C87910(v5, a2, a3, a4, a5);
  sub_2241E40(v5, a2, a3, a4, a5);
  return 0;
}
