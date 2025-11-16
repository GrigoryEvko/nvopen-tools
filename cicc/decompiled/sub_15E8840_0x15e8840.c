// Function: sub_15E8840
// Address: 0x15e8840
//
_QWORD *__fastcall sub_15E8840(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rax

  v8 = sub_15A9620(a2, a1[3], *(_DWORD *)(*(_QWORD *)a3 + 8LL) >> 8);
  v9 = sub_15A0680(v8, a4, 0);
  return sub_15E84D0(a1, a2, a3, v9, a5);
}
