// Function: sub_2D48D60
// Address: 0x2d48d60
//
__int64 __fastcall sub_2D48D60(
        __int64 a1,
        void (__fastcall *a2)(__int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD *, __int64 *, __int64),
        __int64 a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r10
  unsigned __int8 v7; // r9
  unsigned __int16 v8; // cx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v12; // [rsp+8h] [rbp-148h] BYREF
  _QWORD **v13; // [rsp+18h] [rbp-138h] BYREF
  unsigned __int64 v14[2]; // [rsp+20h] [rbp-130h] BYREF
  _BYTE v15[112]; // [rsp+30h] [rbp-120h] BYREF
  void *v16; // [rsp+A0h] [rbp-B0h]
  void *v17; // [rsp+A8h] [rbp-A8h]
  _QWORD v18[10]; // [rsp+100h] [rbp-50h] BYREF

  v12 = (_QWORD *)a1;
  v4 = sub_B43CC0(a1);
  sub_2D46B10((__int64)v14, a1, v4);
  v5 = sub_B43CB0(a1);
  v15[92] = sub_B2D610(v5, 72);
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_BYTE *)(a1 + 72);
  v13 = &v12;
  v8 = *(_WORD *)(a1 + 2);
  _BitScanReverse64(&v9, 1LL << (v8 >> 9));
  v10 = sub_2D460D0(
          (__int64)v14,
          v6,
          *(v12 - 8),
          63 - ((unsigned __int8)v9 ^ 0x3Fu),
          (v8 >> 1) & 7,
          v7,
          sub_2D42E90,
          (__int64)&v13,
          a2,
          a3,
          (__int64)v12);
  sub_BD84D0((__int64)v12, v10);
  sub_B43D60(v12);
  sub_B32BF0(v18);
  v16 = &unk_49E5698;
  v17 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( (_BYTE *)v14[0] != v15 )
    _libc_free(v14[0]);
  return 1;
}
