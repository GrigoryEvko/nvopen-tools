// Function: sub_36CCDB0
// Address: 0x36ccdb0
//
__int64 __fastcall sub_36CCDB0(__int64 a1, int a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // eax
  unsigned int v6; // r8d
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  _QWORD v10[3]; // [rsp+0h] [rbp-20h] BYREF

  v3 = sub_9208B0(a1, a3);
  v10[1] = v4;
  v10[0] = (unsigned __int64)(v3 + 7) >> 3;
  v5 = sub_CA1930(v10);
  v6 = 4;
  v7 = a2 * v5;
  if ( v7 <= 0xF )
  {
    v8 = v7;
    if ( !v7 || (v7 & (v7 - 1)) != 0 )
      v8 = (((((unsigned __int64)v7 >> 1) | v7) >> 2) | ((unsigned __int64)v7 >> 1) | v7) + 1;
    _BitScanReverse64(&v8, v8);
    return 63 - ((unsigned int)v8 ^ 0x3F);
  }
  return v6;
}
