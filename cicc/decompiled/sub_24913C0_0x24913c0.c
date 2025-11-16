// Function: sub_24913C0
// Address: 0x24913c0
//
_BYTE *__fastcall sub_24913C0(const char *a1, __int64 a2, __int64 a3)
{
  size_t v4; // rax
  _BYTE *result; // rax
  _QWORD v6[8]; // [rsp+0h] [rbp-40h] BYREF

  v6[0] = a2;
  v6[1] = a3;
  v6[2] = a1;
  v4 = strlen(a1);
  result = sub_BA8D20(a2, (__int64)a1, v4, a3, (__int64 (__fastcall *)(__int64))sub_2491280, (__int64)v6);
  if ( *result >= 4u )
    return 0;
  return result;
}
