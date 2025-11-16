// Function: sub_168D470
// Address: 0x168d470
//
_QWORD *__fastcall sub_168D470(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD v4[3]; // [rsp+0h] [rbp-20h] BYREF
  _QWORD *v5; // [rsp+18h] [rbp-8h] BYREF

  v4[0] = a2;
  v4[1] = a3;
  v5 = v4;
  return sub_168C980(a1, (void (__fastcall *)(__int64, void **))sub_168C3C0, (__int64)&v5);
}
