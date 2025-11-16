// Function: sub_C11C30
// Address: 0xc11c30
//
__int64 __fastcall sub_C11C30(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 24);
  if ( *(_QWORD *)(a1 + 56) == v1 || !v1 )
    return 0;
  else
    return v1 - 56;
}
