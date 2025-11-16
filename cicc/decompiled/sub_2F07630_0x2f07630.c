// Function: sub_2F07630
// Address: 0x2f07630
//
void *__fastcall sub_2F07630(unsigned int a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  _BYTE v6[16]; // [rsp+0h] [rbp-90h] BYREF
  void (__fastcall *v7)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-80h]
  void (__fastcall *v8)(_BYTE *, _QWORD *); // [rsp+18h] [rbp-78h]
  _QWORD v9[14]; // [rsp+20h] [rbp-70h] BYREF

  v9[5] = 0x100000000LL;
  v9[6] = a2;
  memset(&v9[1], 0, 32);
  v9[0] = &unk_49DD210;
  sub_CB5980((__int64)v9, 0, 0, 0);
  sub_2FF6320(v6, a1, a3, 0, 0);
  if ( !v7 )
    sub_4263D6(v6, a1, v4);
  v8(v6, v9);
  if ( v7 )
    v7(v6, v6, 3);
  v9[0] = &unk_49DD210;
  return sub_CB5840((__int64)v9);
}
