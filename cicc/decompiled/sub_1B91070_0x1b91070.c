// Function: sub_1B91070
// Address: 0x1b91070
//
__int64 __fastcall sub_1B91070(__int64 a1, _QWORD *a2, unsigned __int64 *a3)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rdi
  __int64 result; // rax

  a2[4] = a1;
  v3 = a2[1];
  v4 = *a3;
  a2[2] = a3;
  v4 &= 0xFFFFFFFFFFFFFFF8LL;
  a2[1] = v4 | v3 & 7;
  *(_QWORD *)(v4 + 8) = a2 + 1;
  result = *a3 & 7;
  *a3 = result | (unsigned __int64)(a2 + 1);
  return result;
}
