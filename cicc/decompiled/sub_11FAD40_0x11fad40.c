// Function: sub_11FAD40
// Address: 0x11fad40
//
__int64 __fastcall sub_11FAD40(int a1)
{
  __int64 v1; // rdi

  v1 = (unsigned int)(a1 - 32);
  if ( (unsigned int)v1 > 9 )
    BUG();
  return dword_3F955C0[v1];
}
