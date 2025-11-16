// Function: sub_A086D0
// Address: 0xa086d0
//
_DWORD *__fastcall sub_A086D0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  unsigned __int64 v8; // r9
  _DWORD *result; // rax

  v3 = *a1;
  v4 = *(_DWORD *)a1[1];
  v5 = sub_B9B140(*(_QWORD *)(*a1 + 248), a2, a3);
  sub_A083B0(v3, v5, v4, v6, v7, v8);
  result = (_DWORD *)a1[1];
  ++*result;
  return result;
}
