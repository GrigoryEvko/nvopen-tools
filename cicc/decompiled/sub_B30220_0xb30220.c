// Function: sub_B30220
// Address: 0xb30220
//
__int64 __fastcall sub_B30220(__int64 a1)
{
  __int64 v1; // rcx
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx

  v1 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v2 = *(_QWORD *)(a1 - 8);
    v3 = v2 + v1;
  }
  else
  {
    v2 = a1 - v1;
    v3 = a1;
  }
  for ( ; v2 != v3; v2 += 32 )
  {
    if ( *(_QWORD *)v2 )
    {
      v4 = *(_QWORD *)(v2 + 8);
      **(_QWORD **)(v2 + 16) = v4;
      if ( v4 )
        *(_QWORD *)(v4 + 16) = *(_QWORD *)(v2 + 16);
    }
    *(_QWORD *)v2 = 0;
  }
  return sub_B91E30(a1);
}
