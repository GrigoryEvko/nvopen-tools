// Function: sub_2FC88B0
// Address: 0x2fc88b0
//
__int64 __fastcall sub_2FC88B0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  unsigned int v5; // esi

  v2 = *(_QWORD *)(a1 + 32) + 40LL * a2;
  if ( *(_BYTE *)v2 != 1 )
    return a2 + 1;
  v4 = *(_QWORD *)(v2 + 24);
  if ( v4 == 1 )
    return a2 + 4;
  if ( v4 == 2 )
    return a2 + 2;
  v5 = a2 + 2;
  if ( v4 )
    BUG();
  return v5 + 1;
}
