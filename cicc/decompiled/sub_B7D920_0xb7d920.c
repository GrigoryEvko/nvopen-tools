// Function: sub_B7D920
// Address: 0xb7d920
//
__int64 *__fastcall sub_B7D920(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  _QWORD v7[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v8; // [rsp+18h] [rbp-38h] BYREF
  __int64 v9; // [rsp+20h] [rbp-30h] BYREF
  __int64 v10[5]; // [rsp+28h] [rbp-28h] BYREF

  v2 = *a2;
  *a2 = 0;
  v7[0] = v2 | 1;
  v3 = sub_22077B0(56);
  v4 = v3;
  if ( v3 )
  {
    *(_BYTE *)(v3 + 24) = 0;
    *(_QWORD *)(v3 + 16) = 0;
    *(_DWORD *)(v3 + 40) = 0;
    *(_QWORD *)v3 = &unk_49DA3F8;
    *(_QWORD *)(v3 + 8) = v3 + 24;
    *(_QWORD *)(v3 + 48) = sub_2241E40();
    v8 = v4;
    v5 = v7[0] | 1LL;
    v7[0] = 0;
    v9 = v5;
    v7[1] = 0;
    sub_B7D740(v10, &v9, &v8);
    if ( (v10[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      BUG();
    if ( (v9 & 1) != 0 || (v9 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v9);
    *(_QWORD *)v4 = &unk_49DA448;
  }
  if ( (v7[0] & 1) != 0 || (v7[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(v7);
  *a1 = v4 | 1;
  return a1;
}
