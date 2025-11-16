// Function: sub_C46FB0
// Address: 0xc46fb0
//
__int64 __fastcall sub_C46FB0(_QWORD *a1, int a2)
{
  _QWORD *v2; // rax

  if ( a2 )
  {
    v2 = a1;
    do
    {
      *v2 = ~*v2;
      ++v2;
    }
    while ( v2 != &a1[(unsigned int)(a2 - 1) + 1] );
  }
  return sub_C46200(a1, 1, a2);
}
