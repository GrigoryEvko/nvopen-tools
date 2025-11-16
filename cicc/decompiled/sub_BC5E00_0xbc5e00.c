// Function: sub_BC5E00
// Address: 0xbc5e00
//
__int64 __fastcall sub_BC5E00(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rbx
  unsigned int v5; // r14d
  unsigned int v6; // eax
  _QWORD v8[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v9; // [rsp+20h] [rbp-40h]

  v4 = a1;
  sub_2241E40();
  if ( a3 )
  {
    v5 = 0;
    do
    {
      v8[0] = v4;
      v9 = 260;
      v6 = sub_C823F0(v8, 1);
      if ( v6 )
        v5 = v6;
      v4 += 32;
    }
    while ( v4 != a1 + 32LL * (unsigned int)(a3 - 1) + 32 );
  }
  else
  {
    return 0;
  }
  return v5;
}
