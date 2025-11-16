// Function: sub_223E4D0
// Address: 0x223e4d0
//
__int64 *__fastcall sub_223E4D0(__int64 *a1, const char *a2)
{
  size_t v2; // rax

  if ( a2 )
  {
    v2 = strlen(a2);
    sub_223E0D0(a1, a2, v2);
  }
  else
  {
    sub_222DC80((__int64)a1 + *(_QWORD *)(*a1 - 24), *(_DWORD *)((char *)a1 + *(_QWORD *)(*a1 - 24) + 32) | 1);
  }
  return a1;
}
