// Function: sub_33DD890
// Address: 0x33dd890
//
__int64 __fastcall sub_33DD890(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v8; // ebx
  unsigned int v10; // eax
  unsigned __int64 v11; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v13; // [rsp+20h] [rbp-A0h]
  unsigned int v14; // [rsp+28h] [rbp-98h]
  unsigned __int64 v15; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-88h]
  unsigned __int64 v17; // [rsp+40h] [rbp-80h]
  unsigned int v18; // [rsp+48h] [rbp-78h]
  unsigned __int64 v19; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+58h] [rbp-68h]
  unsigned __int64 v21; // [rsp+60h] [rbp-60h]
  unsigned int v22; // [rsp+68h] [rbp-58h]
  unsigned __int64 v23; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+78h] [rbp-48h]
  unsigned __int64 v25; // [rsp+80h] [rbp-40h]
  unsigned int v26; // [rsp+88h] [rbp-38h]

  v8 = 0;
  if ( !sub_33CF170(a4) )
  {
    sub_33DD090((__int64)&v11, a1, a2, a3, 0);
    sub_33DD090((__int64)&v15, a1, a4, a5, 0);
    sub_AAF050((__int64)&v19, (__int64)&v11, 0);
    sub_AAF050((__int64)&v23, (__int64)&v15, 0);
    v10 = sub_ABE130((__int64)&v19, (__int64)&v23);
    if ( v10 > 3 )
      BUG();
    v8 = dword_44DF830[v10];
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    if ( v20 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    if ( v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    if ( v14 > 0x40 && v13 )
      j_j___libc_free_0_0(v13);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
  }
  return v8;
}
