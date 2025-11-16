// Function: sub_22C1770
// Address: 0x22c1770
//
_QWORD *__fastcall sub_22C1770(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax

  v5 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v6 = sub_B2BEC0(a3);
  a1[2] = 0;
  *a1 = v5 + 8;
  a1[1] = v6;
  return a1;
}
