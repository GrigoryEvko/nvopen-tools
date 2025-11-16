// Function: sub_2A475A0
// Address: 0x2a475a0
//
__int64 __fastcall sub_2A475A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v8; // r14
  char v9; // al
  _QWORD v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2 - a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * (v4 >> 4);
  v6 = a1;
  v11[0] = a4;
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v5 >> 1;
        sub_2A44DC0((__int64)v11, a3, v6 + 16 * ((v5 >> 1) + (v5 & 0xFFFFFFFFFFFFFFFELL)));
        if ( v9 )
          break;
        v6 += 16 * ((v5 >> 1) + (v5 & 0xFFFFFFFFFFFFFFFELL)) + 48;
        v5 = v5 - v8 - 1;
        if ( v5 <= 0 )
          return v6;
      }
      v5 >>= 1;
    }
    while ( v8 > 0 );
  }
  return v6;
}
