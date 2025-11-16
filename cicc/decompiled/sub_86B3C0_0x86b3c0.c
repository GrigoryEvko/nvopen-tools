// Function: sub_86B3C0
// Address: 0x86b3c0
//
__int64 __fastcall sub_86B3C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 result; // rax

  v6 = qword_4F5FD88 + 30;
  v7 = unk_4D03B98 - qword_4F5FD90;
  v8 = sub_822C60((void *)qword_4F5FD90, 176 * (qword_4F5FD88 + 30) - 5280, 176 * (qword_4F5FD88 + 30), a4, a5, a6);
  qword_4F5FD88 = v6;
  qword_4F5FD90 = v8;
  result = v7 + v8;
  unk_4D03B98 = result;
  return result;
}
