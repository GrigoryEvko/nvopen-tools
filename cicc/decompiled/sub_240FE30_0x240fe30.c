// Function: sub_240FE30
// Address: 0x240fe30
//
_QWORD *__fastcall sub_240FE30(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rax
  __int64 v3; // r13
  _QWORD *v4; // rax
  _QWORD *v5; // r12
  __int64 v7; // [rsp+8h] [rbp-58h]
  const char *v8; // [rsp+10h] [rbp-50h] BYREF
  char v9; // [rsp+30h] [rbp-30h]
  char v10; // [rsp+31h] [rbp-2Fh]

  **(_BYTE **)a1 = 1;
  if ( !byte_4FE2E28 && (unsigned int)sub_2207590((__int64)&byte_4FE2E28) )
  {
    byte_4FE2E30 = (_DWORD)qword_4FE2FA8 != 0;
    sub_2207640((__int64)&byte_4FE2E28);
  }
  v1 = 0;
  if ( byte_4FE2E30 )
    v1 = (int)qword_4FE2FA8;
  v2 = sub_ACD640(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL), v1, 1u);
  v10 = 1;
  v3 = v2;
  v9 = 3;
  v8 = "__dfsan_track_origins";
  BYTE4(v7) = 0;
  v4 = sub_BD2C40(88, unk_3F0FAE8);
  v5 = v4;
  if ( v4 )
    sub_B30000(
      (__int64)v4,
      *(_QWORD *)(a1 + 8),
      *(_QWORD **)(*(_QWORD *)(a1 + 16) + 24LL),
      1,
      5,
      v3,
      (__int64)&v8,
      0,
      0,
      v7,
      0);
  return v5;
}
