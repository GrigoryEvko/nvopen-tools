// Function: sub_AB1D50
// Address: 0xab1d50
//
__int64 __fastcall sub_AB1D50(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v4; // rdx
  unsigned int v5; // ebx
  unsigned __int64 v6; // rdi
  unsigned int v7; // ebx
  unsigned int v8; // r12d
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r14
  int v13; // eax
  unsigned int v14; // r12d
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+8h] [rbp-38h]
  unsigned __int64 v18; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-28h]

  v1 = 0;
  if ( sub_AAF7D0(a1) )
    return v1;
  sub_AB13A0((__int64)&v18, a1);
  v4 = 1LL << ((unsigned __int8)v19 - 1);
  if ( v19 > 0x40 )
  {
    v7 = v19 + 1;
    v5 = (*(_QWORD *)(v18 + 8LL * ((v19 - 1) >> 6)) & v4) != 0
       ? v7 - sub_C44500(&v18)
       : v7 - (unsigned int)sub_C444A0(&v18);
  }
  else if ( (v18 & v4) != 0 )
  {
    if ( v19 )
    {
      v5 = v19 - 63;
      if ( v18 << (64 - (unsigned __int8)v19) != -1 )
      {
        _BitScanReverse64(&v6, ~(v18 << (64 - (unsigned __int8)v19)));
        v5 = v19 + 1 - (v6 ^ 0x3F);
      }
    }
    else
    {
      v5 = 1;
    }
  }
  else
  {
    v5 = 1;
    if ( v18 )
    {
      _BitScanReverse64(&v11, v18);
      v5 = 65 - (v11 ^ 0x3F);
    }
  }
  sub_AB14C0((__int64)&v16, a1);
  v8 = v17;
  v9 = 1LL << ((unsigned __int8)v17 - 1);
  if ( v17 > 0x40 )
  {
    v12 = v16;
    if ( (*(_QWORD *)(v16 + 8LL * ((v17 - 1) >> 6)) & v9) != 0 )
      v13 = sub_C44500(&v16);
    else
      v13 = sub_C444A0(&v16);
    v14 = v8 + 1 - v13;
    if ( v14 >= v5 )
      v5 = v14;
    v1 = v5;
    if ( v12 )
      j_j___libc_free_0_0(v12);
  }
  else
  {
    if ( (v16 & v9) != 0 )
    {
      if ( v17 )
      {
        if ( v16 << (64 - (unsigned __int8)v17) == -1 )
        {
          v1 = v17 - 63;
        }
        else
        {
          _BitScanReverse64(&v10, ~(v16 << (64 - (unsigned __int8)v17)));
          v1 = v17 + 1 - (v10 ^ 0x3F);
        }
      }
      else
      {
        v1 = 1;
      }
    }
    else
    {
      v1 = 1;
      if ( v16 )
      {
        _BitScanReverse64(&v15, v16);
        v1 = 65 - (v15 ^ 0x3F);
      }
    }
    if ( v1 < v5 )
      v1 = v5;
  }
  if ( v19 <= 0x40 || !v18 )
    return v1;
  j_j___libc_free_0_0(v18);
  return v1;
}
