// Function: sub_315E260
// Address: 0x315e260
//
__int64 __fastcall sub_315E260(__int64 a1)
{
  __int64 v2; // rax

  if ( *(_BYTE *)a1 > 0x1Cu )
    return *(_QWORD *)(a1 + 32);
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 80LL);
  if ( !v2 )
    BUG();
  return *(_QWORD *)(v2 + 32);
}
