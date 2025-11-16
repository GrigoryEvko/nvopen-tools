// Function: sub_990550
// Address: 0x990550
//
__int64 __fastcall sub_990550(int a1)
{
  __int64 v1; // rdi

  v1 = (unsigned int)(a1 - 1);
  if ( (unsigned int)v1 > 3 )
    BUG();
  return dword_3F1F7F0[v1];
}
