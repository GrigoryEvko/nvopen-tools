// Function: sub_1A246E0
// Address: 0x1a246e0
//
__int64 __fastcall sub_1A246E0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  unsigned int v5; // eax
  __int64 v6; // r9
  __int64 result; // rax
  __int64 v8; // [rsp+8h] [rbp-68h]
  unsigned __int64 v9; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-58h]
  __int128 v11; // [rsp+20h] [rbp-50h]
  __int64 v12; // [rsp+30h] [rbp-40h]

  v4 = a1[16] - a1[7];
  LOWORD(v12) = 257;
  v5 = sub_15A9570(*a1, a3);
  v10 = v5;
  if ( v5 > 0x40 )
    sub_16A4EF0((__int64)&v9, v4, 0);
  else
    v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & v4;
  result = sub_1A23B30(a2, *a1, a1[6], (__int64)&v9, a3, v6, v11, v12);
  if ( v10 > 0x40 )
  {
    if ( v9 )
    {
      v8 = result;
      j_j___libc_free_0_0(v9);
      return v8;
    }
  }
  return result;
}
