// Function: sub_8231B0
// Address: 0x8231b0
//
int *__fastcall sub_8231B0(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int *result; // rax

  sub_823100(a1, a2, a3, a4, a5, a6);
  *((_QWORD *)qword_4F073B0 + a1) = 0;
  result = &dword_4F073A8;
  if ( dword_4F073A8 < a1 )
    dword_4F073A8 = a1;
  return result;
}
