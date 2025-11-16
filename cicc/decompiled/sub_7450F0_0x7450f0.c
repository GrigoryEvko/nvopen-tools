// Function: sub_7450F0
// Address: 0x7450f0
//
__int64 __fastcall sub_7450F0(const char *a1, _DWORD *a2, __int64 (__fastcall **a3)(const char *, _QWORD))
{
  __int64 result; // rax

  if ( *a2 )
    (*a3)(" ", a3);
  (*a3)("__attribute__((", a3);
  (*a3)(a1, a3);
  result = (*a3)("))", a3);
  *a2 = 1;
  return result;
}
