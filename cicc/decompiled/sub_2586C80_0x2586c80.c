// Function: sub_2586C80
// Address: 0x2586c80
//
_BOOL8 __fastcall sub_2586C80(__int64 *a1, __int64 a2)
{
  _BOOL4 v2; // r13d
  unsigned __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rax
  unsigned __int64 v8[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = sub_2585CD0((__int64)a1, a2);
  v3 = sub_250C680(a1 + 9);
  if ( v3 )
  {
    sub_250D230(v8, v3, 6, 0);
    v4 = sub_2584D90(a2, v8[0], v8[1], (__int64)a1, 2, 0, 1);
    if ( v4 )
    {
      v5 = *(_QWORD *)(v4 + 96);
      if ( v5 )
      {
        _BitScanReverse64(&v5, v5);
        v5 = 0x8000000000000000LL >> ((unsigned __int8)v5 ^ 0x3Fu);
      }
      v6 = v5;
      if ( a1[13] >= v5 )
        v6 = a1[13];
      if ( a1[12] >= v5 )
        v5 = a1[12];
      a1[13] = v6;
      a1[12] = v5;
    }
  }
  return v2;
}
