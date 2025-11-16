// Function: sub_1D2C0A0
// Address: 0x1d2c0a0
//
__int64 __fastcall sub_1D2C0A0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rsi
  int v5; // eax
  __int64 *v6; // r15
  __int64 *v7; // rbx
  __int64 *v8; // rdx
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned __int8 v14; // al
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v22; // [rsp+10h] [rbp-80h] BYREF
  int v23; // [rsp+18h] [rbp-78h]
  __int128 v24; // [rsp+20h] [rbp-70h]
  __int64 v25; // [rsp+30h] [rbp-60h]
  __int64 v26; // [rsp+40h] [rbp-50h] BYREF
  __int64 v27; // [rsp+48h] [rbp-48h]
  __int64 v28; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 72);
  v22 = v4;
  if ( v4 )
    sub_1623A60((__int64)&v22, v4, 2);
  v5 = *(_DWORD *)(a2 + 64);
  v6 = *(__int64 **)(a2 + 32);
  LOBYTE(v25) = 0;
  v23 = v5;
  v7 = *(__int64 **)(v6[15] + 88);
  v8 = *(__int64 **)(v6[20] + 88);
  v26 = 0;
  v9 = 0;
  v27 = 0;
  v28 = 0;
  v24 = (unsigned __int64)v8;
  if ( v8 )
  {
    v10 = *v8;
    if ( *(_BYTE *)(*v8 + 8) == 16 )
      v10 = **(_QWORD **)(v10 + 16);
    v9 = *(_DWORD *)(v10 + 8) >> 8;
  }
  v11 = a1[4];
  HIDWORD(v25) = v9;
  v12 = sub_1E0A0C0(v11);
  v13 = 8 * sub_15A9520(v12, 0);
  if ( v13 == 32 )
  {
    v14 = 5;
  }
  else if ( v13 > 0x20 )
  {
    v14 = 6;
    if ( v13 != 64 )
    {
      v14 = 0;
      if ( v13 == 128 )
        v14 = 7;
    }
  }
  else
  {
    v14 = 3;
    if ( v13 != 8 )
      v14 = 4 * (v13 == 16);
  }
  v15 = sub_1D2B730(a1, v14, 0, (__int64)&v22, *v6, v6[1], v6[10], v6[11], v24, v25, 0, 0, (__int64)&v26, 0);
  v26 = 0;
  v17 = v15;
  v18 = 0;
  v24 = (unsigned __int64)v7;
  v27 = 0;
  v28 = 0;
  LOBYTE(v25) = 0;
  if ( v7 )
  {
    v19 = *v7;
    if ( *(_BYTE *)(*v7 + 8) == 16 )
      v19 = **(_QWORD **)(v19 + 16);
    v18 = *(_DWORD *)(v19 + 8) >> 8;
  }
  HIDWORD(v25) = v18;
  v20 = sub_1D2BF40(
          a1,
          v17,
          1,
          (__int64)&v22,
          v17,
          v16,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
          v24,
          v25,
          0,
          0,
          (__int64)&v26);
  if ( v22 )
    sub_161E7C0((__int64)&v22, v22);
  return v20;
}
