// Function: sub_12FB6F0
// Address: 0x12fb6f0
//
unsigned __int64 __fastcall sub_12FB6F0(__int128 a1, unsigned __int64 a2)
{
  if ( a2 <= 0x3F )
    return (*((_QWORD *)&a1 + 1) << (-(char)a2 & 0x3F) != 0)
         | (*((_QWORD *)&a1 + 1) >> a2)
         | ((_QWORD)a1 << (-(char)a2 & 0x3F));
  if ( a2 > 0x7E )
    return a1 != 0;
  return ((unsigned __int64)a1 >> a2) | ((*((_QWORD *)&a1 + 1) | (unsigned __int64)a1 & ~(-1LL << a2)) != 0);
}
