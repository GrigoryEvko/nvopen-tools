// Function: sub_E6B340
// Address: 0xe6b340
//
_QWORD *__fastcall sub_E6B340(_QWORD *a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r12

  v1 = a1[108];
  a1[118] += 152LL;
  v2 = (_QWORD *)((v1 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( a1[109] >= (unsigned __int64)(v2 + 19) && v1 )
    a1[108] = v2 + 19;
  else
    v2 = (_QWORD *)sub_9D1E70((__int64)(a1 + 108), 152, 152, 3);
  sub_E92760((_DWORD)v2, 6, (unsigned int)byte_3F871B3, 0, 1, 0, 0);
  *v2 = &unk_49E1A38;
  sub_E6B260(a1, (__int64)v2);
  return v2;
}
