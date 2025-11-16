// Function: sub_18EE330
// Address: 0x18ee330
//
__int64 __fastcall sub_18EE330(__int64 ***a1, int a2, int a3)
{
  unsigned int v4; // r12d
  __int64 v6; // [rsp+0h] [rbp-90h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-88h]
  __int64 v8; // [rsp+10h] [rbp-80h]
  unsigned int v9; // [rsp+18h] [rbp-78h]
  __int64 v10; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v11; // [rsp+28h] [rbp-68h]
  __int64 v12; // [rsp+30h] [rbp-60h]
  unsigned int v13; // [rsp+38h] [rbp-58h]
  __int64 v14; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+48h] [rbp-48h]
  __int64 v16; // [rsp+50h] [rbp-40h]
  unsigned int v17; // [rsp+58h] [rbp-38h]

  sub_13F2950((__int64)&v6, *a1[1], (**a1)[3 * (1LL - (*((_DWORD *)**a1 + 5) & 0xFFFFFFF))], (**a1)[5], (__int64)**a1);
  v4 = 0;
  sub_1591060((__int64)&v10, a2, (__int64)&v6, a3);
  if ( !sub_158A120((__int64)&v10) )
  {
    sub_13F2950((__int64)&v14, *a1[1], (**a1)[-3 * (*((_DWORD *)**a1 + 5) & 0xFFFFFFF)], (**a1)[5], (__int64)**a1);
    v4 = sub_158BB40((__int64)&v10, (__int64)&v14);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  return v4;
}
