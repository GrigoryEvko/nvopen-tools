// Function: sub_222DFB0
// Address: 0x222dfb0
//
__int64 __fastcall sub_222DFB0(__int64 a1, unsigned int a2)
{
  __int64 **i; // rbx
  __int64 result; // rax

  for ( i = *(__int64 ***)(a1 + 40); i; i = (__int64 **)*i )
    result = ((__int64 (__fastcall *)(_QWORD, __int64, _QWORD))i[1])(a2, a1, *((unsigned int *)i + 4));
  return result;
}
