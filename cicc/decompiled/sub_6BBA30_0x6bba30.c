// Function: sub_6BBA30
// Address: 0x6bba30
//
__int64 __fastcall sub_6BBA30(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // r12
  __int64 result; // rax
  __int64 v4; // [rsp+8h] [rbp-C8h] BYREF
  _BYTE v5[192]; // [rsp+10h] [rbp-C0h] BYREF

  v1 = qword_4F06BC0;
  if ( unk_4F07270 == unk_4F073B8 && qword_4F06BC0 && (*(_BYTE *)(qword_4F06BC0 - 8LL) & 1) == 0 )
    qword_4F06BC0 = *(_QWORD *)(unk_4F07288 + 88LL);
  sub_6E2250(v5, &v4, 4, 1, a1, 0);
  v2 = sub_6BB5A0(1, 0);
  sub_6E6470(v2);
  sub_6E1990(v2);
  sub_6E2A90();
  result = sub_6E2C70(v4, 1, a1, 0);
  if ( a1 )
    *(_BYTE *)(a1 + 127) &= ~4u;
  qword_4F06BC0 = v1;
  return result;
}
