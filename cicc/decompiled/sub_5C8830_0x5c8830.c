// Function: sub_5C8830
// Address: 0x5c8830
//
_BYTE *__fastcall sub_5C8830(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax

  if ( a3 != 11 )
    return (_BYTE *)a2;
  if ( (*(_QWORD *)(a2 + 200) & 0x8000001000000LL) == 0x8000000000000LL )
  {
    v4 = sub_8258E0(a2, 0);
    sub_6865F0(3469, a1 + 56, "__global__", v4);
    return (_BYTE *)a2;
  }
  return sub_5C8600(a1, a2);
}
