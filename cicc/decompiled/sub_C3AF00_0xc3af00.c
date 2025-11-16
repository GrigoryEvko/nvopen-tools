// Function: sub_C3AF00
// Address: 0xc3af00
//
unsigned int *__fastcall sub_C3AF00(__int64 a1, _DWORD *a2, __int64 *a3)
{
  if ( a2 == (_DWORD *)&unk_3F65800 )
    return sub_C38180(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F657E0 )
    return sub_C38000(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F657C0 )
    return sub_C37E70(a1, (__int64)a3);
  if ( a2 == dword_3F657A0 )
    return sub_C37CF0(a1, (unsigned __int64)a3);
  if ( a2 == (_DWORD *)&unk_3F655E0 )
    return sub_C373D0(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F65780 )
    return sub_C37B40(a1, a3);
  if ( a2 == dword_3F65580 )
    return (unsigned int *)sub_C3AE00(a1, a3);
  if ( a2 == (_DWORD *)&unk_3F65760 )
    return sub_C38310(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F65740 )
    return (unsigned int *)sub_C38480(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F65720 )
    return sub_C385D0(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F65700 )
    return (unsigned int *)sub_C38740(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F656E0 )
    return (unsigned int *)sub_C388A0(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F656C0 )
    return (unsigned int *)sub_C389F0(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F656A0 )
    return sub_C38B40(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F65680 )
    return sub_C38CB0(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F65660 )
    return (unsigned int *)sub_C36E40(a1, a3);
  if ( a2 == (_DWORD *)&unk_3F65640 )
    return (unsigned int *)sub_C38E40(a1, (__int64)a3);
  if ( a2 == (_DWORD *)&unk_3F65620 )
    return (unsigned int *)sub_C38F50(a1, (__int64)a3);
  if ( a2 != (_DWORD *)&unk_3F65600 )
    BUG();
  return (unsigned int *)sub_C39060(a1, (__int64)a3);
}
