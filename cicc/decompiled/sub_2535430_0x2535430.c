// Function: sub_2535430
// Address: 0x2535430
//
unsigned __int64 __fastcall sub_2535430(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 16) <= a2 )
    a2 = *(_QWORD *)(a1 + 16);
  if ( a2 < result )
    a2 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = a2;
  return result;
}
