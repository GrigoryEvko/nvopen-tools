// Function: sub_A719A0
// Address: 0xa719a0
//
__int64 __fastcall sub_A719A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v7; // rbx
  __int64 v8; // r14
  unsigned __int64 *v9; // rdi

  v7 = (unsigned __int64 *)(a2 + 64);
  v8 = a2 + 64 + 8LL * *(unsigned int *)(a2 + 8);
  if ( a2 + 64 != v8 )
  {
    do
    {
      v9 = v7++;
      sub_A718C0(v9, a5);
    }
    while ( (unsigned __int64 *)v8 != v7 );
  }
  return sub_C656C0(a5, a3);
}
