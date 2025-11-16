// Function: sub_BC9770
// Address: 0xbc9770
//
__int64 __fastcall sub_BC9770(__int64 a1, const char *a2, _QWORD *a3)
{
  int v4; // eax
  _BYTE *v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rax

  if ( !a1 )
    return 0;
  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
    v4 = *(_DWORD *)(a1 - 24);
  else
    v4 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  if ( v4 != 2 )
    return 0;
  v5 = sub_BC96B0(a1, a2);
  if ( !v5 )
    return 0;
  v6 = *((_QWORD *)v5 + 17);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  *a3 = v7;
  return 1;
}
