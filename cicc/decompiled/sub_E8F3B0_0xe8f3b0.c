// Function: sub_E8F3B0
// Address: 0xe8f3b0
//
void __fastcall sub_E8F3B0(__int64 a1, __int64 a2)
{
  unsigned int *v2; // rbx
  unsigned int *v3; // rdi

  if ( a2 - a1 <= 384 )
  {
    sub_E8F2F0(a1, a2);
  }
  else
  {
    v2 = (unsigned int *)(a1 + 384);
    sub_E8F2F0(a1, a1 + 384);
    if ( a2 != a1 + 384 )
    {
      do
      {
        v3 = v2;
        v2 += 6;
        sub_E8F2A0(v3);
      }
      while ( (unsigned int *)a2 != v2 );
    }
  }
}
