// Function: sub_30CD8C0
// Address: 0x30cd8c0
//
void __fastcall sub_30CD8C0(__int64 a1)
{
  __int64 v1; // rsi
  unsigned __int8 *v2; // r8
  unsigned __int8 *v3; // rcx
  __int64 v4; // r15
  unsigned __int8 *v5; // [rsp-58h] [rbp-58h]
  unsigned __int8 *v6; // [rsp-50h] [rbp-50h]
  char v7; // [rsp-41h] [rbp-41h] BYREF
  __int64 v8[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a1 + 56) )
  {
    v1 = *(_QWORD *)(a1 + 32);
    v2 = *(unsigned __int8 **)(a1 + 16);
    v3 = *(unsigned __int8 **)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 40);
    v8[0] = v1;
    if ( v1 )
    {
      v5 = v3;
      v6 = v2;
      sub_B96E90((__int64)v8, v1, 1);
      v3 = v5;
      v2 = v6;
    }
    sub_30CD350(
      *(__int64 **)(a1 + 48),
      v8,
      v4,
      v3,
      v2,
      1,
      (void (__fastcall *)(__int64, _QWORD *))sub_30CA240,
      (__int64)&v7,
      0);
    if ( v8[0] )
      sub_B91220((__int64)v8, v8[0]);
  }
}
