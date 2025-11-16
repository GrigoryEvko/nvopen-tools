// Function: sub_C45D30
// Address: 0xc45d30
//
void __fastcall sub_C45D30(__int64 a1, __int64 a2, int a3)
{
  __int64 i; // rax

  if ( a3 )
  {
    for ( i = 0; i != a3; ++i )
      *(_QWORD *)(a1 + 8 * i) = *(_QWORD *)(a2 + 8 * i);
  }
}
