// Function: sub_228ACA0
// Address: 0x228aca0
//
__int64 __fastcall sub_228ACA0(unsigned __int64 a1)
{
  __int64 *v2; // rbx
  __int64 v3; // r13
  unsigned int v4; // r12d
  __int64 v5; // rdi

  if ( (a1 & 1) != 0 )
    return (int)sub_39FAC40(~(-1LL << (a1 >> 58)) & (a1 >> 1));
  v2 = *(__int64 **)a1;
  v3 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 == v3 )
    return 0;
  v4 = 0;
  do
  {
    v5 = *v2++;
    v4 += sub_39FAC40(v5);
  }
  while ( (__int64 *)v3 != v2 );
  return v4;
}
