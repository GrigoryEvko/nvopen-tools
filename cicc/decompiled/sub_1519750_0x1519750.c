// Function: sub_1519750
// Address: 0x1519750
//
_DWORD *__fastcall sub_1519750(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned int v4; // r13d
  __int64 v5; // rax
  _DWORD *result; // rax

  v3 = *a1;
  v4 = *(_DWORD *)a1[1];
  v5 = sub_161FF10(*(_QWORD *)(*a1 + 240), a2, a3);
  sub_15194E0(v3, v5, v4);
  result = (_DWORD *)a1[1];
  ++*result;
  return result;
}
