// Function: sub_1442F90
// Address: 0x1442f90
//
__int64 __fastcall sub_1442F90(__int64 a1)
{
  __int64 v1; // rax
  unsigned int i; // r8d

  v1 = *(_QWORD *)(a1 + 8);
  for ( i = 0; v1; ++i )
    v1 = *(_QWORD *)(v1 + 8);
  return i;
}
