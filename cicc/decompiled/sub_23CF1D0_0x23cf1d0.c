// Function: sub_23CF1D0
// Address: 0x23cf1d0
//
__int64 __fastcall sub_23CF1D0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(unsigned int *)(a1 + 636);
  if ( (unsigned int)v1 > 4 )
    BUG();
  return qword_437F2C0[v1];
}
