// Function: sub_2E78D90
// Address: 0x2e78d90
//
__int64 __fastcall sub_2E78D90(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // r8
  __int64 result; // rax

  v3 = a1[1];
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 32LL);
  result = 0;
  if ( v4 == sub_23CE290 )
  {
    a1[5] = 0;
  }
  else
  {
    result = ((__int64 (__fastcall *)(__int64, _QWORD *, _QWORD, __int64))v4)(v3, a1 + 16, *a1, a2);
    a1[5] = result;
  }
  return result;
}
