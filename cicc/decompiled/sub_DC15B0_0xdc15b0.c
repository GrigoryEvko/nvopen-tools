// Function: sub_DC15B0
// Address: 0xdc15b0
//
__int64 __fastcall sub_DC15B0(__int64 a1, int a2, __int64 a3, __int64 a4, int a5, __int64 a6, __int64 a7)
{
  unsigned int v7; // r14d
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // [rsp+10h] [rbp-C0h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-B8h]
  __int64 v16; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-A8h]
  unsigned __int8 v18; // [rsp+30h] [rbp-A0h]
  __int64 v19; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v20; // [rsp+48h] [rbp-88h]
  __int64 v21; // [rsp+50h] [rbp-80h]
  unsigned int v22; // [rsp+58h] [rbp-78h]
  __int64 v23; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+68h] [rbp-68h]
  __int64 v25; // [rsp+70h] [rbp-60h]
  unsigned int v26; // [rsp+78h] [rbp-58h]
  __int64 v27; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+88h] [rbp-48h]
  __int64 v29; // [rsp+90h] [rbp-40h]
  unsigned int v30; // [rsp+98h] [rbp-38h]

  v7 = 0;
  if ( !*(_WORD *)(a4 + 24) && !*(_WORD *)(a7 + 24) )
  {
    sub_DC06D0((__int64)&v16, a1, a3, a6);
    v7 = v18;
    if ( v18 )
    {
      sub_AB1A50((__int64)&v19, a5, *(_QWORD *)(a7 + 32) + 24LL);
      v15 = v17;
      if ( v17 > 0x40 )
        sub_C43780((__int64)&v14, (const void **)&v16);
      else
        v14 = v16;
      sub_AADBC0((__int64)&v27, &v14);
      sub_AB4F10((__int64)&v23, (__int64)&v19, (__int64)&v27);
      if ( v30 > 0x40 && v29 )
        j_j___libc_free_0_0(v29);
      if ( v28 > 0x40 && v27 )
        j_j___libc_free_0_0(v27);
      if ( v15 > 0x40 && v14 )
        j_j___libc_free_0_0(v14);
      v12 = *(_QWORD *)(a4 + 32);
      v15 = *(_DWORD *)(v12 + 32);
      if ( v15 > 0x40 )
        sub_C43780((__int64)&v14, (const void **)(v12 + 24));
      else
        v14 = *(_QWORD *)(v12 + 24);
      sub_AADBC0((__int64)&v27, &v14);
      LOBYTE(v13) = sub_ABB410(&v23, a2, &v27);
      v7 = v13;
      if ( v30 > 0x40 && v29 )
        j_j___libc_free_0_0(v29);
      if ( v28 > 0x40 && v27 )
        j_j___libc_free_0_0(v27);
      if ( v15 > 0x40 && v14 )
        j_j___libc_free_0_0(v14);
      if ( v26 > 0x40 && v25 )
        j_j___libc_free_0_0(v25);
      if ( v24 > 0x40 && v23 )
        j_j___libc_free_0_0(v23);
      if ( v22 > 0x40 && v21 )
        j_j___libc_free_0_0(v21);
      if ( v20 > 0x40 && v19 )
        j_j___libc_free_0_0(v19);
      if ( v18 )
      {
        v18 = 0;
        if ( v17 > 0x40 )
        {
          if ( v16 )
            j_j___libc_free_0_0(v16);
        }
      }
    }
  }
  return v7;
}
