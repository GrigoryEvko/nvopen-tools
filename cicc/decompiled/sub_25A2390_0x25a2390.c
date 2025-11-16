// Function: sub_25A2390
// Address: 0x25a2390
//
__int64 __fastcall sub_25A2390(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  _BYTE *v3; // r14
  unsigned __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v9; // [rsp+18h] [rbp-2D8h] BYREF
  _QWORD v10[4]; // [rsp+20h] [rbp-2D0h] BYREF
  _QWORD v11[2]; // [rsp+40h] [rbp-2B0h] BYREF
  __int16 v12; // [rsp+50h] [rbp-2A0h]
  __int64 v13; // [rsp+58h] [rbp-298h]
  __int64 v14; // [rsp+60h] [rbp-290h]
  __int64 v15; // [rsp+68h] [rbp-288h]
  __int64 v16; // [rsp+70h] [rbp-280h]
  _BYTE *v17; // [rsp+78h] [rbp-278h]
  __int64 v18; // [rsp+80h] [rbp-270h]
  _BYTE v19[136]; // [rsp+88h] [rbp-268h] BYREF
  _BYTE v20[208]; // [rsp+110h] [rbp-1E0h] BYREF
  _BYTE v21[272]; // [rsp+1E0h] [rbp-110h] BYREF

  v10[1] = a2;
  v13 = 0;
  v14 = 0;
  v11[0] = &unk_4A170B8;
  v12 = 256;
  v15 = 0;
  v17 = v19;
  v11[1] = &unk_4A16CD8;
  v18 = 0x800000000LL;
  memset(v21, 0, 0xD8u);
  v16 = 0;
  v19[128] = 0;
  v9 = 0;
  v10[0] = &v9;
  v10[2] = a1;
  v10[3] = v21;
  if ( !(unsigned __int8)sub_2527330(
                           a2,
                           (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2584B00,
                           (__int64)v10,
                           a1,
                           1u,
                           1u) )
  {
    HIBYTE(v12) = v12;
    if ( !v21[208] )
      goto LABEL_3;
    goto LABEL_23;
  }
  if ( v21[208] )
  {
    if ( !v21[17] )
      HIBYTE(v12) = v12;
    sub_2576560((__int64)v11, (__int64)v21);
    sub_2560F70((__int64)v20, (__int64)v11);
    sub_25485A0((__int64)v20);
    if ( v21[208] )
    {
LABEL_23:
      v21[208] = 0;
      sub_25485A0((__int64)v21);
    }
  }
LABEL_3:
  v2 = sub_25A1B90(a1 + 88, (__int64)v11);
  v3 = v17;
  v11[0] = &unk_4A170B8;
  v4 = (unsigned __int64)&v17[16 * (unsigned int)v18];
  if ( v17 != (_BYTE *)v4 )
  {
    do
    {
      v4 -= 16LL;
      if ( *(_DWORD *)(v4 + 8) > 0x40u && *(_QWORD *)v4 )
        j_j___libc_free_0_0(*(_QWORD *)v4);
    }
    while ( v3 != (_BYTE *)v4 );
    v4 = (unsigned __int64)v17;
  }
  if ( (_BYTE *)v4 != v19 )
    _libc_free(v4);
  v5 = (unsigned int)v16;
  if ( (_DWORD)v16 )
  {
    v6 = v14;
    v7 = v14 + 16LL * (unsigned int)v16;
    do
    {
      if ( *(_DWORD *)(v6 + 8) > 0x40u && *(_QWORD *)v6 )
        j_j___libc_free_0_0(*(_QWORD *)v6);
      v6 += 16;
    }
    while ( v7 != v6 );
    v5 = (unsigned int)v16;
  }
  sub_C7D6A0(v14, 16 * v5, 8);
  return v2;
}
