// Function: sub_19D1240
// Address: 0x19d1240
//
__int64 __fastcall sub_19D1240(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rbx
  __int64 v7; // rdi
  unsigned int v8; // r15d
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax

  v3 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 24 * (2 - v3)) + 16LL) != 13 )
    return 0;
  v7 = *(_QWORD *)(a2 + 24 * (3 - v3));
  v8 = *(_DWORD *)(v7 + 32);
  if ( v8 <= 0x40 )
  {
    if ( *(_QWORD *)(v7 + 24) )
      return 0;
  }
  else if ( v8 != (unsigned int)sub_16A57B0(v7 + 24) )
  {
    return 0;
  }
  v9 = *(_QWORD *)(a2 + 24 * (1 - v3));
  v10 = sub_1649C60(*(_QWORD *)(a2 - 24 * v3));
  v11 = sub_19D0490(a1, a2, v10, v9);
  if ( !v11 )
    return 0;
  *a3 = v11 + 24;
  return 1;
}
