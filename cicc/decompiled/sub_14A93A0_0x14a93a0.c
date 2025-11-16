// Function: sub_14A93A0
// Address: 0x14a93a0
//
unsigned __int64 *__fastcall sub_14A93A0(unsigned __int64 *a1, __int64 a2, __int64 *a3, unsigned int a4)
{
  unsigned int v5; // eax
  unsigned __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // rdx

  v5 = *((_DWORD *)a3 + 2);
  *((_DWORD *)a1 + 2) = v5;
  if ( v5 > 0x40 )
  {
    sub_16A4FD0(a1, a3);
    v5 = *((_DWORD *)a1 + 2);
    if ( v5 > 0x40 )
    {
      sub_16A5E70(a1, a4);
      return a1;
    }
    v6 = *a1;
  }
  else
  {
    v6 = *a3;
  }
  v7 = (__int64)(v6 << (64 - (unsigned __int8)v5)) >> (64 - (unsigned __int8)v5);
  v8 = v7 >> a4;
  v9 = v7 >> 63;
  if ( a4 == v5 )
    v8 = v9;
  v10 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & v8;
  *a1 = v10;
  return a1;
}
