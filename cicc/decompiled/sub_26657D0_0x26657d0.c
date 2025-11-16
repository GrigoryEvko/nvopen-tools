// Function: sub_26657D0
// Address: 0x26657d0
//
__int64 __fastcall sub_26657D0(_QWORD *a1)
{
  __int64 v1; // rsi
  __int64 result; // rax
  unsigned __int64 v3[2]; // [rsp+8h] [rbp-38h] BYREF
  __int64 v4; // [rsp+18h] [rbp-28h]
  __int64 v5; // [rsp+20h] [rbp-20h]

  v1 = a1[1];
  v3[1] = 0;
  v3[0] = v1 & 6;
  v4 = a1[3];
  result = v4;
  if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
  {
    sub_BD6050(v3, v1 & 0xFFFFFFFFFFFFFFF8LL);
    result = v4;
    v5 = a1[4];
    if ( v4 != 0 && v4 != -4096 && v4 != -8192 )
      return sub_BD60C0(v3);
  }
  return result;
}
