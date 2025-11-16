// Function: sub_B130B0
// Address: 0xb130b0
//
__int64 __fastcall sub_B130B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = sub_B11FB0(a1 + 40);
  v2 = *(_QWORD *)(a1 + 24);
  v3 = v1;
  v7[0] = v2;
  if ( v2 )
    sub_B96E90(v7, v2, 1);
  v4 = sub_22077B0(48);
  v5 = v4;
  if ( v4 )
    sub_B12570(v4, v3, v7);
  if ( v7[0] )
    sub_B91220(v7);
  return v5;
}
