// Function: sub_1111170
// Address: 0x1111170
//
char __fastcall sub_1111170(unsigned int **a1, char a2, char a3)
{
  unsigned int v4; // eax
  __int64 v5; // rsi
  unsigned __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 v8; // rsi
  unsigned __int64 v9; // rsi
  char result; // al
  char v11; // [rsp+Fh] [rbp-41h]
  __int64 v12; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-38h]
  unsigned __int64 v14; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-28h]

  v13 = **a1;
  if ( v13 > 0x40 )
    sub_C43690((__int64)&v12, 0, 0);
  else
    v12 = 0;
  if ( a2 )
  {
    v4 = **a1;
    v5 = 2LL * (**(_BYTE **)a1[1] == 68) - 1;
    v15 = v4;
    if ( v4 > 0x40 )
    {
      sub_C43690((__int64)&v14, v5, 1);
    }
    else
    {
      v6 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & v5;
      if ( !v4 )
        v6 = 0;
      v14 = v6;
    }
    sub_C45EE0((__int64)&v12, (__int64 *)&v14);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  if ( a3 )
  {
    v7 = **a1;
    v8 = 2LL * (**(_BYTE **)a1[2] == 68) - 1;
    v15 = v7;
    if ( v7 > 0x40 )
    {
      sub_C43690((__int64)&v14, v8, 1);
    }
    else
    {
      v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & v8;
      if ( !v7 )
        v9 = 0;
      v14 = v9;
    }
    sub_C45EE0((__int64)&v12, (__int64 *)&v14);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  result = sub_B532C0((__int64)&v12, a1[3], *a1[4]);
  if ( v13 > 0x40 )
  {
    if ( v12 )
    {
      v11 = result;
      j_j___libc_free_0_0(v12);
      return v11;
    }
  }
  return result;
}
