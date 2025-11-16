// Function: sub_C14100
// Address: 0xc14100
//
__int64 __fastcall sub_C14100(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 48) == v1 )
    return 0;
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(v1 + 8);
  return 1;
}
