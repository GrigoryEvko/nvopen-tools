// Function: sub_A71960
// Address: 0xa71960
//
__int64 __fastcall sub_A71960(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v3; // rbx
  __int64 result; // rax
  __int64 v5; // r13
  unsigned __int64 *v7; // rdi

  v3 = (unsigned __int64 *)(a2 + 64);
  result = *(unsigned int *)(a2 + 8);
  v5 = a2 + 64 + 8 * result;
  if ( a2 + 64 != v5 )
  {
    do
    {
      v7 = v3++;
      result = sub_A718C0(v7, a3);
    }
    while ( (unsigned __int64 *)v5 != v3 );
  }
  return result;
}
