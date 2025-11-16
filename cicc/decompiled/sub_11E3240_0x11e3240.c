// Function: sub_11E3240
// Address: 0x11e3240
//
__int64 __fastcall sub_11E3240(__int64 a1, __int64 a2, unsigned int **a3)
{
  __int64 *v4; // r15
  __int64 v5; // r12
  __int64 v7; // rax

  v4 = (__int64 *)sub_B43CA0(a2);
  v5 = sub_11E3150(a1, a2, a3);
  if ( !v5 && sub_11C99B0(v4, *(__int64 **)(a1 + 24), 0xBAu) )
  {
    if ( (unsigned __int8)sub_988330(a2) )
    {
      v7 = sub_11CA9C0(
             *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
             *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
             *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
             (__int64)a3,
             *(_QWORD *)(a1 + 16),
             *(__int64 **)(a1 + 24));
      if ( v7 )
      {
        v5 = v7;
        if ( *(_BYTE *)v7 == 85 )
          *(_WORD *)(v7 + 2) = *(_WORD *)(v7 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
      }
    }
  }
  return v5;
}
