// Function: sub_2C1C370
// Address: 0x2c1c370
//
__int64 *__fastcall sub_2C1C370(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v9; // [rsp+8h] [rbp-78h]
  _QWORD v10[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v11[4]; // [rsp+20h] [rbp-60h] BYREF
  char v12; // [rsp+40h] [rbp-40h]
  char v13; // [rsp+41h] [rbp-3Fh]

  v11[0] = *(_QWORD *)(a1 + 88);
  if ( v11[0] )
    sub_2AAAFA0(v11);
  sub_2BF1A90(a2, (__int64)v11);
  sub_9C6650(v11);
  v3 = *(_QWORD *)(a2 + 904);
  v4 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
  v5 = sub_2BFB640(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL), 0);
  BYTE4(v9) = 0;
  v6 = *(_QWORD *)(v5 + 8);
  v10[1] = v4;
  v11[0] = (__int64)"partial.reduce";
  v13 = 1;
  v12 = 3;
  v10[0] = v5;
  v7 = sub_B35180(v3, v6, 0xA3u, (__int64)v10, 2u, v9, (__int64)v11);
  return sub_2BF26E0(a2, a1 + 96, v7, 0);
}
