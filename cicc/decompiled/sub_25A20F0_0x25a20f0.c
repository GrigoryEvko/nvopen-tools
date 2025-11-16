// Function: sub_25A20F0
// Address: 0x25a20f0
//
__int64 __fastcall sub_25A20F0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  _BYTE *v3; // r14
  unsigned __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 v7; // rbx
  char v9; // [rsp+1Bh] [rbp-2D5h] BYREF
  int v10; // [rsp+1Ch] [rbp-2D4h] BYREF
  _QWORD v11[4]; // [rsp+20h] [rbp-2D0h] BYREF
  _QWORD v12[2]; // [rsp+40h] [rbp-2B0h] BYREF
  __int16 v13; // [rsp+50h] [rbp-2A0h]
  __int64 v14; // [rsp+58h] [rbp-298h]
  __int64 v15; // [rsp+60h] [rbp-290h]
  __int64 v16; // [rsp+68h] [rbp-288h]
  __int64 v17; // [rsp+70h] [rbp-280h]
  _BYTE *v18; // [rsp+78h] [rbp-278h]
  __int64 v19; // [rsp+80h] [rbp-270h]
  _BYTE v20[136]; // [rsp+88h] [rbp-268h] BYREF
  _BYTE v21[208]; // [rsp+110h] [rbp-1E0h] BYREF
  _BYTE v22[272]; // [rsp+1E0h] [rbp-110h] BYREF

  v14 = 0;
  v15 = 0;
  v18 = v20;
  v12[0] = &unk_4A170B8;
  v13 = 256;
  v16 = 0;
  v20[128] = 0;
  v12[1] = &unk_4A16CD8;
  v19 = 0x800000000LL;
  memset(v22, 0, 0xD8u);
  v17 = 0;
  v10 = sub_250CB50((__int64 *)(a1 + 72), 0);
  v11[0] = &v10;
  v11[1] = a2;
  v11[2] = a1;
  v11[3] = v22;
  v9 = 0;
  if ( !(unsigned __int8)sub_2523890(
                           a2,
                           (__int64 (__fastcall *)(__int64, __int64 *))sub_2584810,
                           (__int64)v11,
                           a1,
                           1u,
                           &v9) )
  {
    HIBYTE(v13) = v13;
    if ( !v22[208] )
      goto LABEL_3;
    goto LABEL_23;
  }
  if ( v22[208] )
  {
    if ( !v22[17] )
      HIBYTE(v13) = v13;
    sub_2576560((__int64)v12, (__int64)v22);
    sub_2560F70((__int64)v21, (__int64)v12);
    sub_25485A0((__int64)v21);
    if ( v22[208] )
    {
LABEL_23:
      v22[208] = 0;
      sub_25485A0((__int64)v22);
    }
  }
LABEL_3:
  v2 = sub_25A1B90(a1 + 88, (__int64)v12);
  v3 = v18;
  v12[0] = &unk_4A170B8;
  v4 = (unsigned __int64)&v18[16 * (unsigned int)v19];
  if ( v18 != (_BYTE *)v4 )
  {
    do
    {
      v4 -= 16LL;
      if ( *(_DWORD *)(v4 + 8) > 0x40u && *(_QWORD *)v4 )
        j_j___libc_free_0_0(*(_QWORD *)v4);
    }
    while ( v3 != (_BYTE *)v4 );
    v4 = (unsigned __int64)v18;
  }
  if ( (_BYTE *)v4 != v20 )
    _libc_free(v4);
  v5 = (unsigned int)v17;
  if ( (_DWORD)v17 )
  {
    v6 = v15;
    v7 = v15 + 16LL * (unsigned int)v17;
    do
    {
      if ( *(_DWORD *)(v6 + 8) > 0x40u && *(_QWORD *)v6 )
        j_j___libc_free_0_0(*(_QWORD *)v6);
      v6 += 16;
    }
    while ( v7 != v6 );
    v5 = (unsigned int)v17;
  }
  sub_C7D6A0(v15, 16 * v5, 8);
  return v2;
}
