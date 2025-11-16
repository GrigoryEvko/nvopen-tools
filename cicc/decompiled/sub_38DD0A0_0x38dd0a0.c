// Function: sub_38DD0A0
// Address: 0x38dd0a0
//
unsigned __int64 __fastcall sub_38DD0A0(__int64 *a1, __int64 a2, unsigned __int8 a3)
{
  void (*v4)(); // r13
  unsigned __int64 result; // rax

  v4 = *(void (**)())(*a1 + 496);
  result = sub_38CB470(a2, a1[1]);
  if ( v4 != nullsub_1954 )
    return ((__int64 (__fastcall *)(__int64 *, unsigned __int64, _QWORD, _QWORD))v4)(a1, result, a3, 0);
  return result;
}
