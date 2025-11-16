// Function: sub_134B960
// Address: 0x134b960
//
__int64 __fastcall sub_134B960(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rsi
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  unsigned int v7; // eax
  char v8; // cl
  unsigned int v9; // eax
  __int64 v10; // rsi

  v2 = *(_QWORD *)(a2 + 104);
  v4 = (__int64)(a1 + 522);
  if ( v2 )
  {
    v4 = (__int64)(a1 + 516);
    if ( v2 != 512 )
    {
      v5 = sub_130F9A0(*(_QWORD *)(a2 + 96) << 12);
      if ( v5 > 0x7000000000000000LL )
      {
        v10 = 10608;
      }
      else
      {
        _BitScanReverse64(&v6, v5);
        v7 = v6 - ((((v5 - 1) & v5) == 0) - 1);
        if ( v7 < 0xE )
          v7 = 14;
        v8 = v7 - 3;
        v9 = v7 - 14;
        if ( !v9 )
          v8 = 12;
        v10 = 48 * ((((v5 - 1) >> v8) & 3) + 4 * v9) + 1056;
      }
      v4 = (__int64)a1 + v10;
    }
  }
  return sub_134B3B0(a1, v4, a2);
}
