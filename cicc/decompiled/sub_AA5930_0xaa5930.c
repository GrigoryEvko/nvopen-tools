// Function: sub_AA5930
// Address: 0xaa5930
//
__int64 __fastcall sub_AA5930(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r8

  if ( a1 + 48 == (*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  v1 = *(_QWORD *)(a1 + 56);
  if ( !v1 )
    BUG();
  v2 = 0;
  if ( *(_BYTE *)(v1 - 24) == 84 )
    return v1 - 24;
  return v2;
}
