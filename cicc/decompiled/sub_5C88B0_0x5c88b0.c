// Function: sub_5C88B0
// Address: 0x5c88b0
//
_BYTE *__fastcall sub_5C88B0(__int64 a1, __int64 a2, char a3)
{
  _BYTE *v3; // r12
  _BYTE *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax

  v3 = (_BYTE *)a2;
  if ( a3 == 7 )
    return sub_5C6B80(a1, (_BYTE *)a2, 7);
  if ( a3 != 11 )
    return v3;
  if ( (*(_QWORD *)(a2 + 200) & 0x8000001000000LL) != 0x8000000000000LL || (*(_BYTE *)(a2 + 192) & 2) != 0 )
  {
    if ( !unk_4D045EC && (*(_BYTE *)(a2 + 198) & 0x20) != 0 )
      sub_6851C0(3481, a1 + 56);
    *(_BYTE *)(a2 + 198) |= 0x11u;
    v5 = sub_5C6B80(a1, (_BYTE *)a2, 11);
  }
  else
  {
    v5 = (_BYTE *)a2;
    v7 = sub_8258E0(a2, 0);
    sub_6865F0(3469, a1 + 56, "__device__", v7);
  }
  if ( (*(_BYTE *)(a1 + 11) & 1) != 0 )
  {
    v6 = *(_QWORD *)(a2 + 152);
    v3 = v5;
    if ( v6 )
    {
      while ( *(_BYTE *)(v6 + 140) == 12 )
        v6 = *(_QWORD *)(v6 + 160);
      v3 = v5;
      sub_5C6D70(**(_QWORD ***)(v6 + 168), *(_QWORD *)(a1 + 56));
    }
    return v3;
  }
  return v5;
}
