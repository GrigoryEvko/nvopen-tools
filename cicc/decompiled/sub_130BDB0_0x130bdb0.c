// Function: sub_130BDB0
// Address: 0x130bdb0
//
__int64 __fastcall sub_130BDB0(
        __int64 a1,
        _QWORD *a2,
        unsigned int *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  unsigned int v13; // r15d
  __int64 result; // rax

  v13 = *a3;
  if ( (unsigned __int8)sub_133DEF0(a1, a2 + 7, 1, *a3, 1) )
    return 1;
  if ( (unsigned __int8)sub_133DEF0(a1, a2 + 2437, 2, v13, 0) )
    return 1;
  if ( (unsigned __int8)sub_133DEF0(a1, a2 + 4867, 3, v13, 0) )
    return 1;
  sub_1343130(a2 + 7300);
  if ( (unsigned __int8)sub_130AF40((__int64)(a2 + 7301)) )
    return 1;
  a2[7330] = a7;
  if ( (unsigned __int8)sub_133D910(a2 + 7331, a6, a8) )
    return 1;
  if ( (unsigned __int8)sub_133D910(a2 + 7554, a6, a9) )
    return 1;
  result = sub_130AF40((__int64)(a2 + 7315));
  if ( (_BYTE)result )
    return 1;
  a2[7329] = 0;
  a2[7297] = a3;
  a2[7298] = a4;
  a2[7778] = a10;
  a2[7299] = a5;
  a2[7777] = a11;
  a2[7779] = 0;
  *a2 = sub_130C1E0;
  a2[2] = sub_130BAA0;
  a2[3] = sub_130B9F0;
  a2[4] = sub_130B960;
  a2[1] = sub_130D0F0;
  a2[6] = sub_130BC50;
  a2[5] = sub_130D1D0;
  return result;
}
