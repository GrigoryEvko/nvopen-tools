// Function: sub_C20E50
// Address: 0xc20e50
//
unsigned __int64 __fastcall sub_C20E50(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  unsigned __int64 i; // r8

  v1 = *(_QWORD *)(a1 + 400);
  v2 = *(_QWORD *)(a1 + 408);
  for ( i = 0; v2 != v1; v1 += 40 )
  {
    if ( i < *(_QWORD *)(v1 + 16) + *(_QWORD *)(v1 + 24) )
      i = *(_QWORD *)(v1 + 16) + *(_QWORD *)(v1 + 24);
  }
  return i;
}
