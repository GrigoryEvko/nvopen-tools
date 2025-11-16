// Function: sub_BD2DD0
// Address: 0xbd2dd0
//
__int64 __fastcall sub_BD2DD0(__int64 a1)
{
  char v1; // al
  __int64 v2; // rsi
  __int64 v3; // r13

  v1 = *(_BYTE *)(a1 + 7);
  v2 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (v1 & 0x40) != 0 )
  {
    sub_BD2950(*(_QWORD *)(a1 - 8), *(_QWORD *)(a1 - 8) + v2, 1);
    return j___libc_free_0(a1 - 8);
  }
  else
  {
    v3 = a1 - v2;
    if ( v1 < 0 )
    {
      sub_BD2950(v3, a1, 0);
      return j___libc_free_0(v3 - *(_QWORD *)(v3 - 8) - 8);
    }
    else
    {
      sub_BD2950(v3, a1, 0);
      return j___libc_free_0(v3);
    }
  }
}
