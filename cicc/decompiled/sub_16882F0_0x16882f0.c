// Function: sub_16882F0
// Address: 0x16882f0
//
int __fastcall sub_16882F0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  v2 = (_QWORD *)a1[2];
  if ( v2 )
  {
    sub_1683BE0(v2, (__int64 (__fastcall *)(_QWORD, __int64))sub_1688230, 0);
    sub_1683B30((_QWORD *)a1[2]);
  }
  return sub_16856A0(a1);
}
