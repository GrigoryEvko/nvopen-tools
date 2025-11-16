// Function: sub_1022EF0
// Address: 0x1022ef0
//
__int64 __fastcall sub_1022EF0(int a1)
{
  __int64 v1; // rdi

  v1 = (unsigned int)(a1 - 1);
  if ( (unsigned int)v1 > 0x13 )
    BUG();
  return dword_3F8DA80[v1];
}
