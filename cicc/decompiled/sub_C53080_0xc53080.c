// Function: sub_C53080
// Address: 0xc53080
//
_QWORD *__fastcall sub_C53080(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v5; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v7[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a1 + 13) & 0x40) != 0 )
  {
    if ( !qword_4F83CE0 )
      sub_C7D570(&qword_4F83CE0, sub_C53DA0, sub_C50EC0);
    v7[0] = &v5;
    v5 = a1;
    v6[0] = a2;
    v6[1] = a3;
    v7[1] = v6;
    v7[2] = qword_4F83CE0;
    result = sub_C52DD0(qword_4F83CE0, a1, (__int64 (__fastcall *)(__int64, __int64))sub_C521F0, (__int64)v7);
  }
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 32) = a3;
  if ( a3 == 1 )
    *(_BYTE *)(a1 + 13) |= 0x10u;
  return result;
}
