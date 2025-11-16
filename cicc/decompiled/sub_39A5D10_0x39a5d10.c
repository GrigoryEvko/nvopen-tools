// Function: sub_39A5D10
// Address: 0x39a5d10
//
__int64 __fastcall sub_39A5D10(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  _DWORD v5[7]; // [rsp-1Ch] [rbp-1Ch] BYREF

  result = a1[27];
  if ( !result )
  {
    v2 = sub_39A5A90((__int64)a1, 36, (__int64)(a1 + 1), 0);
    a1[27] = v2;
    sub_39A3F30(a1, v2, 3, "__ARRAY_SIZE_TYPE__", 0x13u);
    v3 = a1[27];
    BYTE2(v5[0]) = 0;
    sub_39A3560((__int64)a1, (__int64 *)(v3 + 8), 11, (__int64)v5, 8);
    v4 = a1[27];
    v5[0] = 65547;
    sub_39A3560((__int64)a1, (__int64 *)(v4 + 8), 62, (__int64)v5, 7);
    sub_3990480(a1[25], (__int64)"__ARRAY_SIZE_TYPE__", 19, a1[27]);
    return a1[27];
  }
  return result;
}
