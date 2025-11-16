// Function: sub_127CA80
// Address: 0x127ca80
//
_QWORD *__fastcall sub_127CA80(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // rdi
  unsigned __int64 v3; // rax
  unsigned __int64 v5[2]; // [rsp+0h] [rbp-10h] BYREF

  v2 = (_QWORD *)(a1 + 72);
  v3 = *(v2 - 7);
  v5[0] = a2;
  v5[1] = v3;
  return sub_91CD50(v2, v5);
}
