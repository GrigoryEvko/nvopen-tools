// Function: sub_A4F4F0
// Address: 0xa4f4f0
//
__int64 __fastcall sub_A4F4F0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 24) == v1 )
    return 0;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(v1 + 8);
  return 1;
}
