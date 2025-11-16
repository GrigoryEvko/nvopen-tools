// Function: sub_851FD0
// Address: 0x851fd0
//
unsigned __int8 *__fastcall sub_851FD0(unsigned __int8 *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  _QWORD *v6; // rdx

  if ( !qword_4D044D8 || sub_7215C0(a1) )
    return a1;
  v6 = (_QWORD *)qword_4F5F860;
  if ( !qword_4F5F860 )
  {
    qword_4F5F860 = (__int64)sub_8237A0(256, a2, 0, v2, v3, v4);
    v6 = (_QWORD *)qword_4F5F860;
  }
  sub_720CF0(qword_4D044D8, (const char *)a1, v6);
  return *(unsigned __int8 **)(qword_4F5F860 + 32);
}
