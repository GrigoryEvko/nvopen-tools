// Function: sub_B6E220
// Address: 0xb6e220
//
__int64 __fastcall sub_B6E220(int a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  int *v6[2]; // [rsp+0h] [rbp-B0h] BYREF
  int *v7; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v8; // [rsp+18h] [rbp-98h]
  _BYTE v9[144]; // [rsp+20h] [rbp-90h] BYREF

  v3 = 0;
  if ( a1 )
  {
    v8 = 0x800000000LL;
    v7 = (int *)v9;
    sub_B6DAB0(a1, (__int64)&v7);
    v6[0] = v7;
    v6[1] = (int *)(unsigned int)v8;
    if ( !(unsigned int)sub_B6B020(a2, v6, a3) )
      v3 = sub_B6B160(*(_DWORD *)(a2 + 8) >> 8 != 0, (__int64 *)v6) ^ 1;
    if ( v7 != (int *)v9 )
      _libc_free(v7, v6);
  }
  return v3;
}
