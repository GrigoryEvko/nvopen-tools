// Function: sub_D00320
// Address: 0xd00320
//
__int64 __fastcall sub_D00320(__int64 **a1, __int64 a2)
{
  __int64 v3; // rax

  if ( *(_BYTE *)a2 > 0x1Cu )
    return sub_98CF40(**a1, a2, *a1[1], 1);
  if ( *(_BYTE *)a2 == 22 )
  {
    v3 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 80LL);
    if ( !v3 )
      BUG();
    a2 = *(_QWORD *)(v3 + 32);
    if ( a2 )
      a2 -= 24;
    return sub_98CF40(**a1, a2, *a1[1], 1);
  }
  return 0;
}
