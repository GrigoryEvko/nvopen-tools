// Function: sub_1688B00
// Address: 0x1688b00
//
__int64 __fastcall sub_1688B00(__int64 a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // rbx

  result = sub_1688AF0();
  if ( (_BYTE)result )
  {
    v2 = qword_4F9F3A8 + a1;
    qword_4F9F3A8 = v2;
    if ( v2 >= qword_4F9F3B0 )
      qword_4F9F3B0 = v2;
  }
  return result;
}
