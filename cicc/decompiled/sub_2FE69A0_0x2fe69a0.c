// Function: sub_2FE69A0
// Address: 0x2fe69a0
//
__int64 __fastcall sub_2FE69A0(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  unsigned int v7; // r12d
  unsigned int v8; // ebx
  unsigned int v9; // eax
  unsigned int v11; // eax
  unsigned int v12; // eax
  char v13; // [rsp+Ch] [rbp-84h]
  unsigned __int64 v14; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-78h]
  unsigned __int64 v16; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-68h]
  unsigned __int64 v18; // [rsp+30h] [rbp-60h]
  unsigned int v19; // [rsp+38h] [rbp-58h]
  unsigned __int64 v20; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+48h] [rbp-48h]
  unsigned __int64 v22; // [rsp+50h] [rbp-40h]
  unsigned int v23; // [rsp+58h] [rbp-38h]

  v13 = BYTE4(a3);
  v21 = 64;
  v20 = (unsigned int)a3;
  sub_AADBC0((__int64)&v16, (__int64 *)&v20);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v13 )
  {
    sub_ABA9E0((__int64)&v20, (__int64)&v16, a5);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    v16 = v20;
    v12 = v21;
    v21 = 0;
    v17 = v12;
    if ( v19 > 0x40 && v18 )
    {
      j_j___libc_free_0_0(v18);
      v18 = v22;
      v19 = v23;
      if ( v21 > 0x40 && v20 )
        j_j___libc_free_0_0(v20);
    }
    else
    {
      v18 = v22;
      v19 = v23;
    }
  }
  if ( a4 )
  {
    v15 = 64;
    v14 = 1;
    sub_AB1F90((__int64)&v20, (__int64 *)&v16, (__int64)&v14);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    v16 = v20;
    v11 = v21;
    v21 = 0;
    v17 = v11;
    if ( v19 > 0x40 && v18 )
    {
      j_j___libc_free_0_0(v18);
      v18 = v22;
      v19 = v23;
      if ( v21 > 0x40 && v20 )
        j_j___libc_free_0_0(v20);
    }
    else
    {
      v18 = v22;
      v19 = v23;
    }
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  v7 = 8;
  v8 = sub_BCB060(a2);
  v9 = sub_AB1CA0((__int64)&v16);
  if ( v8 > v9 )
    v8 = v9;
  if ( v8 > 1 )
  {
    _BitScanReverse(&v8, v8 - 1);
    v7 = 1 << (32 - (v8 ^ 0x1F));
    if ( v7 < 8 )
      v7 = 8;
  }
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return v7;
}
