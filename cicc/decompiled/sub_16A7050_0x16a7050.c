// Function: sub_16A7050
// Address: 0x16a7050
//
void __fastcall sub_16A7050(__int64 a1, __int64 a2, int a3)
{
  __int64 i; // rax

  if ( a3 )
  {
    for ( i = 0; i != a3; ++i )
      *(_QWORD *)(a1 + 8 * i) = *(_QWORD *)(a2 + 8 * i);
  }
}
