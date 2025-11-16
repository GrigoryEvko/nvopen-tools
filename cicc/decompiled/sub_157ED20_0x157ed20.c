// Function: sub_157ED20
// Address: 0x157ed20
//
__int64 __fastcall sub_157ED20(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi

  v1 = *(_QWORD *)(a1 + 48);
  v2 = a1 + 40;
  if ( v1 == v2 )
    return 0;
  while ( 1 )
  {
    if ( !v1 )
      BUG();
    if ( *(_BYTE *)(v1 - 8) != 77 )
      break;
    v1 = *(_QWORD *)(v1 + 8);
    if ( v2 == v1 )
      return 0;
  }
  return v1 - 24;
}
