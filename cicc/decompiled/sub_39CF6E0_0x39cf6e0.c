// Function: sub_39CF6E0
// Address: 0x39cf6e0
//
void *__fastcall sub_39CF6E0(unsigned int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v5[2]; // [rsp+0h] [rbp-60h] BYREF
  void (__fastcall *v6)(__int64 *, __int64 *, __int64); // [rsp+10h] [rbp-50h]
  void (__fastcall *v7)(__int64 *, __int64 *); // [rsp+18h] [rbp-48h]
  __int64 v8[4]; // [rsp+20h] [rbp-40h] BYREF
  int v9; // [rsp+40h] [rbp-20h]
  __int64 v10; // [rsp+48h] [rbp-18h]

  v10 = a2;
  v9 = 1;
  memset(&v8[1], 0, 24);
  v8[0] = (__int64)&unk_49EFBE0;
  sub_1F4AA00(v5, a1, a3, 0, 0);
  if ( !v6 )
    sub_4263D6(v5, a1, v3);
  v7(v5, v8);
  if ( v6 )
    v6(v5, v5, 3);
  return sub_16E7BC0(v8);
}
