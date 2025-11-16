// Function: sub_7218F0
// Address: 0x7218f0
//
void *__fastcall sub_7218F0(FILE *a1, __int64 a2, int a3, __off_t a4, size_t a5, void *a6, __int64 a7)
{
  int v11; // eax
  void *result; // rax
  int *v13; // rax

  v11 = fileno(a1);
  result = mmap(a6, a5, a3 == 0 ? 3 : 1, a6 == 0 ? 2 : 18, v11, a4);
  if ( result == (void *)-1LL || a6 != result && a6 != 0 || !result )
  {
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F077C8;
    v13 = __errno_location();
    sub_686660(0x6B2u, a7, *v13);
  }
  return result;
}
