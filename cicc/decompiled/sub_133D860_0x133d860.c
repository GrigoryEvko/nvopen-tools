// Function: sub_133D860
// Address: 0x133d860
//
__int64 __fastcall sub_133D860(__int64 *a1, __int64 *a2, __int64 a3)
{
  a1[15] = a3;
  if ( a3 > 0 )
  {
    sub_130B0C0(a1 + 16, 1000000 * a3);
    sub_130B210((unsigned __int64 *)a1 + 16, 0xC8u);
  }
  sub_130B140(a1 + 17, a2);
  a1[18] = (__int64)a1;
  sub_133D760(a1);
  a1[21] = 0;
  a1[22] = 0;
  a1[221] = 0;
  memset(
    (void *)((unsigned __int64)(a1 + 23) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 184) & 0xFFFFFFF8) + 1776) >> 3));
  return 0;
}
