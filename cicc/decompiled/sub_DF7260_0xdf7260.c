// Function: sub_DF7260
// Address: 0xdf7260
//
__int64 __fastcall sub_DF7260(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8)
{
  __int64 v8; // r10
  int v9; // r9d
  _QWORD *v10; // r8
  __int64 result; // rax
  _QWORD *v12; // r11
  int v13; // edx

  v8 = a3;
  v9 = a4;
  if ( a2 <= 0x18 )
  {
    result = 4;
    if ( a2 > 0x12 )
      return result;
  }
  else if ( a2 - 28 <= 1 )
  {
    v10 = sub_DF7050(a7, (__int64)&a7[a8]);
    result = 0;
    if ( v12 != v10 )
      return result;
  }
  result = 1;
  if ( v9 == 1 )
  {
    v13 = *(unsigned __int8 *)(v8 + 8);
    if ( (unsigned int)(v13 - 17) <= 1 )
      LOBYTE(v13) = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
    result = 3;
    if ( (unsigned __int8)v13 > 3u && (_BYTE)v13 != 5 )
      return 2LL * ((v13 & 0xFD) == 4) + 1;
  }
  return result;
}
