// Function: sub_226F2F0
// Address: 0x226f2f0
//
_QWORD *__fastcall sub_226F2F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  _QWORD v7[8]; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD v8[12]; // [rsp+40h] [rbp-60h] BYREF

  sub_CF6F50(v7, a2 + 8, a3, a4);
  sub_CF4B80(v8, v7);
  v4 = (_QWORD *)sub_22077B0(0x40u);
  v5 = v4;
  if ( v4 )
  {
    *v4 = &unk_4A08588;
    sub_CF4B80(v4 + 1, v8);
  }
  sub_CF4BF0(v8);
  *a1 = v5;
  sub_CF4BF0(v7);
  return a1;
}
