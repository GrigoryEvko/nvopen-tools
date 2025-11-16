// Function: sub_C52F90
// Address: 0xc52f90
//
_QWORD *__fastcall sub_C52F90(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD v5[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v6[8]; // [rsp+10h] [rbp-40h] BYREF

  if ( !qword_4F83CE0 )
    sub_C7D570(&qword_4F83CE0, sub_C53DA0, sub_C50EC0);
  v5[0] = a2;
  v5[1] = a3;
  v6[0] = a1;
  v6[1] = v5;
  v6[2] = qword_4F83CE0;
  return sub_C52DD0(qword_4F83CE0, a1, (__int64 (__fastcall *)(__int64, __int64))sub_C521D0, (__int64)v6);
}
