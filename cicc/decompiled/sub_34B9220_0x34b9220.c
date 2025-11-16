// Function: sub_34B9220
// Address: 0x34b9220
//
__int64 __fastcall sub_34B9220(int a1)
{
  __int64 v1; // rdi

  v1 = (unsigned int)(a1 - 32);
  if ( (unsigned int)v1 > 9 )
    BUG();
  return dword_44E23C0[v1];
}
