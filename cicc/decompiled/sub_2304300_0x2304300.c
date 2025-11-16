// Function: sub_2304300
// Address: 0x2304300
//
_QWORD *__fastcall sub_2304300(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  _QWORD *v5; // rax
  _QWORD v7[3]; // [rsp+8h] [rbp-18h] BYREF

  sub_DF45E0(v7, a2 + 8, a3, a4);
  v4 = v7[0];
  v5 = (_QWORD *)sub_22077B0(0x10u);
  if ( v5 )
  {
    v5[1] = v4;
    *v5 = &unk_4A15A50;
  }
  *a1 = v5;
  return a1;
}
