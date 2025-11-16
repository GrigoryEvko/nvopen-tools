// Function: sub_C860A0
// Address: 0xc860a0
//
__suseconds_t __fastcall sub_C860A0(_QWORD *a1, _QWORD *a2, __suseconds_t *a3)
{
  __suseconds_t result; // rax
  struct rusage usage; // [rsp+0h] [rbp-B0h] BYREF

  *a1 = sub_220F850();
  getrusage(RUSAGE_SELF, &usage);
  result = 1000 * (usage.ru_stime.tv_usec + 1000000 * usage.ru_stime.tv_sec);
  *a2 = 1000 * (usage.ru_utime.tv_usec + 1000000 * usage.ru_utime.tv_sec);
  *a3 = result;
  return result;
}
