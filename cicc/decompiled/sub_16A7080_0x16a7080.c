// Function: sub_16A7080
// Address: 0x16a7080
//
__int64 __fastcall sub_16A7080(_QWORD *a1, int a2)
{
  __int64 v2; // rax

  if ( !a2 )
    return 1;
  v2 = (__int64)&a1[(unsigned int)(a2 - 1) + 1];
  while ( !*a1 )
  {
    if ( ++a1 == (_QWORD *)v2 )
      return 1;
  }
  return 0;
}
