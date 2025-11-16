// Function: sub_C14190
// Address: 0xc14190
//
__int64 __fastcall sub_C14190(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 24);
  if ( *(_QWORD *)(a1 + 56) == v1 || !v1 )
    return 0;
  else
    return v1 - 56;
}
