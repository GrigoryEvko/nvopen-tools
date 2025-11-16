// Function: sub_2FDCE70
// Address: 0x2fdce70
//
_QWORD *__fastcall sub_2FDCE70(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx

  v3 = *(_QWORD *)(a3 + 56);
  *a1 = a1 + 2;
  a1[2] = v3;
  a1[3] = a3 + 48;
  a1[1] = 0x300000001LL;
  return a1;
}
