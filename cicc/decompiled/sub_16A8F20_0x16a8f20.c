// Function: sub_16A8F20
// Address: 0x16a8f20
//
void __fastcall sub_16A8F20(_QWORD *a1, int a2)
{
  __int64 v2; // rax

  if ( a2 )
  {
    v2 = (__int64)&a1[(unsigned int)(a2 - 1) + 1];
    do
    {
      *a1 = ~*a1;
      ++a1;
    }
    while ( a1 != (_QWORD *)v2 );
  }
}
