// Function: sub_28B0630
// Address: 0x28b0630
//
__int64 __fastcall sub_28B0630(__int64 a1, __int64 a2, _WORD *a3)
{
  __int64 v5; // rdi
  unsigned int v6; // ebx

  v5 = *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 <= 0x40 )
  {
    if ( *(_QWORD *)(v5 + 24) )
      return 0;
  }
  else if ( v6 != (unsigned int)sub_C444A0(v5 + 24) )
  {
    return 0;
  }
  return sub_28AFC20(a1, a2, a3);
}
