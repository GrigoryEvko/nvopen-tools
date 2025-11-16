// Function: sub_67C320
// Address: 0x67c320
//
__int64 __fastcall sub_67C320(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v5; // rdx

  v1 = 0;
  if ( (*(_DWORD *)(a1 + 176) & 0x11000) != 0x1000 )
    return v1;
  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  v2 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  v1 = *(_QWORD *)(v2 + 104);
  if ( v1 )
    return v1;
  v4 = *(_QWORD *)(*(_QWORD *)(v2 + 72) + 88LL);
  v5 = *(_QWORD *)(v4 + 88);
  if ( v5 )
  {
    if ( (*(_BYTE *)(v4 + 160) & 1) == 0 )
      v4 = *(_QWORD *)(v5 + 88);
  }
  return *(_QWORD *)(v4 + 176);
}
