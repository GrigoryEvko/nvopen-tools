// Function: sub_880E90
// Address: 0x880e90
//
__int64 __fastcall sub_880E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  __int64 v7; // rdi
  __int64 v9; // r13
  __int64 v10; // rax

  v6 = dword_4F04C30;
  if ( dword_4F04C30 == 0x7FFFFFFF )
    sub_685240(0x8Fu);
  ++dword_4F04C30;
  v7 = (__int64)qword_4F066A0;
  if ( v6 >= qword_4F5FFC0 )
  {
    v9 = qword_4F5FFC0 + 0x4000;
    v10 = sub_822C60(qword_4F066A0, 8 * (qword_4F5FFC0 + 0x4000) - 0x20000, 8 * (qword_4F5FFC0 + 0x4000), a4, a5, a6);
    qword_4F5FFC0 = v9;
    qword_4F066A0 = (void *)v10;
    v7 = v10;
  }
  *(_QWORD *)(v7 + 8LL * v6) = qword_4D03FF0;
  return (unsigned int)v6;
}
