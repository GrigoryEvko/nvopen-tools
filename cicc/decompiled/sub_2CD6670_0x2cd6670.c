// Function: sub_2CD6670
// Address: 0x2cd6670
//
__int64 __fastcall sub_2CD6670(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rbx
  __int64 *v4; // rax
  __int64 v5; // r12
  unsigned __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v10; // [rsp+8h] [rbp-E8h] BYREF
  const char *v11; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v12; // [rsp+18h] [rbp-D8h]
  char *v13; // [rsp+20h] [rbp-D0h]
  __int16 v14; // [rsp+30h] [rbp-C0h]
  unsigned int *v15[2]; // [rsp+40h] [rbp-B0h] BYREF
  char v16; // [rsp+50h] [rbp-A0h] BYREF
  void *v17; // [rsp+C0h] [rbp-30h]

  v10 = a1;
  v2 = (__int64 *)sub_BD5C60(a1);
  v3 = sub_BCE3C0(v2, 0);
  sub_23D0AB0((__int64)v15, a2, 0, 0, 0);
  v11 = (const char *)v3;
  v12 = *(_QWORD *)(a1 + 8);
  v4 = (__int64 *)sub_B43CA0(a2);
  v5 = sub_B6E160(v4, 0x22DBu, (__int64)&v11, 2);
  v11 = sub_BD5D20(a1);
  v14 = 773;
  v6 = 0;
  v12 = v7;
  v13 = ".gen";
  if ( v5 )
    v6 = *(_QWORD *)(v5 + 24);
  v8 = sub_921880(v15, v6, v5, (int)&v10, 1, (__int64)&v11, 0);
  nullsub_61();
  v17 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v15[0] != &v16 )
    _libc_free((unsigned __int64)v15[0]);
  return v8;
}
