// Function: sub_A77160
// Address: 0xa77160
//
unsigned __int64 __fastcall sub_A77160(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v4; // rbx
  __int64 v5; // r13
  unsigned __int64 *v6; // rdi

  v4 = (unsigned __int64 *)(a2 + 64);
  v5 = a2 + 64 + 8LL * *(unsigned int *)(a2 + 8);
  if ( a2 + 64 != v5 )
  {
    do
    {
      v6 = v4++;
      sub_A718C0(v6, a3);
    }
    while ( (unsigned __int64 *)v5 != v4 );
  }
  return sub_939680(*(_QWORD **)a3, *(_QWORD *)a3 + 4LL * *(unsigned int *)(a3 + 8));
}
