// Function: sub_5C8110
// Address: 0x5c8110
//
_BYTE *__fastcall sub_5C8110(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax

  if ( a3 != 11 )
    return (_BYTE *)a2;
  if ( (*(_QWORD *)(a2 + 200) & 0x8000001000000LL) == 0x8000000000000LL && (*(_BYTE *)(a2 + 192) & 2) == 0 )
  {
    v4 = sub_8258E0(a2, 0);
    sub_6865F0(3469, a1 + 56, "__host__", v4);
    return (_BYTE *)a2;
  }
  if ( (*(_BYTE *)(a2 + 198) & 0x20) != 0 )
    sub_6851C0(3481, a1 + 56);
  *(_BYTE *)(a2 + 198) |= 0xAu;
  return sub_5C6B80(a1, (_BYTE *)a2, 11);
}
