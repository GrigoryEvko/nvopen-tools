// Function: sub_AA5D10
// Address: 0xaa5d10
//
__int64 __fastcall sub_AA5D10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rdi

  if ( a2 != a4 )
  {
    v5 = a2;
    do
    {
      v6 = v5;
      v5 = *(_QWORD *)(v5 + 8);
      sub_B43D60(v6 - 24, a2, a3, a4);
    }
    while ( a4 != v5 );
  }
  return a4;
}
