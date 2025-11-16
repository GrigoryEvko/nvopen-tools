// Function: sub_C53010
// Address: 0xc53010
//
_QWORD *__fastcall sub_C53010(__int64 a1)
{
  __int64 v2; // [rsp+8h] [rbp-28h] BYREF
  _QWORD v3[3]; // [rsp+10h] [rbp-20h] BYREF

  if ( !qword_4F83CE0 )
    sub_C7D570(&qword_4F83CE0, sub_C53DA0, sub_C50EC0);
  v2 = a1;
  v3[0] = &v2;
  v3[1] = qword_4F83CE0;
  return sub_C52DD0(qword_4F83CE0, a1, (__int64 (__fastcall *)(__int64, __int64))sub_C50950, (__int64)v3);
}
