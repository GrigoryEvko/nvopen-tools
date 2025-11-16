// Function: sub_F518D0
// Address: 0xf518d0
//
__int64 __fastcall sub_F518D0(
        unsigned __int8 *a1,
        unsigned int a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned int v7; // r15d
  unsigned int v10; // r15d
  unsigned __int64 v11; // rax
  unsigned int v12; // r15d
  __int64 result; // rax
  unsigned __int8 v14; // [rsp+Eh] [rbp-52h]
  unsigned __int8 v15; // [rsp+Eh] [rbp-52h]
  __int64 v16; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-48h]
  __int64 v18; // [rsp+20h] [rbp-40h]
  unsigned int v19; // [rsp+28h] [rbp-38h]

  sub_9AC3E0((__int64)&v16, (__int64)a1, a3, 0, a5, a4, a6, 1);
  v7 = v17;
  if ( v17 > 0x40 )
  {
    LODWORD(_RCX) = sub_C445E0((__int64)&v16);
    if ( (unsigned int)_RCX > 0x20 )
      LODWORD(_RCX) = 32;
  }
  else
  {
    LODWORD(_RCX) = 32;
    _RAX = ~v16;
    if ( v16 != -1 )
    {
      __asm { tzcnt   rcx, rax }
      if ( (unsigned int)_RCX > 0x20 )
        LODWORD(_RCX) = 32;
    }
  }
  v10 = v7 - 1;
  if ( v10 <= (unsigned int)_RCX )
    LOBYTE(_RCX) = v10;
  _BitScanReverse64(&v11, 1LL << _RCX);
  v12 = 63 - (v11 ^ 0x3F);
  result = v12;
  if ( BYTE1(a2) )
  {
    if ( (unsigned __int8)v12 < (unsigned __int8)a2 )
    {
      result = sub_F51790(a1, a2, a3);
      if ( (unsigned __int8)v12 >= (unsigned __int8)result )
        result = v12;
    }
  }
  if ( v19 > 0x40 && v18 )
  {
    v14 = result;
    j_j___libc_free_0_0(v18);
    result = v14;
  }
  if ( v17 > 0x40 )
  {
    if ( v16 )
    {
      v15 = result;
      j_j___libc_free_0_0(v16);
      return v15;
    }
  }
  return result;
}
