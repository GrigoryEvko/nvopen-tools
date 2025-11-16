// Function: sub_30603E0
// Address: 0x30603e0
//
_QWORD *__fastcall sub_30603E0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  _QWORD *result; // rax

  v2 = *a2;
  a2[10] += 280;
  result = (_QWORD *)((v2 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( a2[1] >= (unsigned __int64)(result + 35) && v2 )
    *a2 = (__int64)(result + 35);
  else
    result = (_QWORD *)sub_9D1E70((__int64)a2, 280, 280, 3);
  result[2] = 0x800000000LL;
  *result = &unk_4A30878;
  result[1] = result + 3;
  return result;
}
