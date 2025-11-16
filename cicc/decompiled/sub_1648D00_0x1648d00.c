// Function: sub_1648D00
// Address: 0x1648d00
//
__int64 __fastcall sub_1648D00(__int64 a1, int a2)
{
  __int64 v2; // rax

  v2 = *(_QWORD *)(a1 + 8);
  if ( !a2 )
    return 1;
  while ( v2 )
  {
    v2 = *(_QWORD *)(v2 + 8);
    if ( !--a2 )
      return 1;
  }
  return 0;
}
