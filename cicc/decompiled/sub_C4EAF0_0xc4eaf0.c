// Function: sub_C4EAF0
// Address: 0xc4eaf0
//
__int64 __fastcall sub_C4EAF0(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  unsigned int v4; // r13d
  __int64 v6; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-58h]
  __int64 v8; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v9; // [rsp+28h] [rbp-48h]
  __int64 v10; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+38h] [rbp-38h]

  v4 = 2 * a2[2];
  sub_C44830((__int64)&v6, a2, v4);
  sub_C44830((__int64)&v8, a3, v4);
  sub_C472A0((__int64)&v10, (__int64)&v6, &v8);
  sub_C440A0(a1, &v10, a2[2], a2[2]);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  return a1;
}
