// Function: sub_11A11E0
// Address: 0x11a11e0
//
void __fastcall sub_11A11E0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rax

  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  else
    v3 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v4 = 32LL * a2 + v3;
  if ( *(_QWORD *)v4 )
  {
    v5 = *(_QWORD *)(v4 + 8);
    **(_QWORD **)(v4 + 16) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v4 + 16);
  }
  *(_QWORD *)v4 = a3;
  if ( a3 )
  {
    v6 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v4 + 8) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = v4 + 8;
    *(_QWORD *)(v4 + 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = v4;
  }
}
