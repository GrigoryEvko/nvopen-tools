// Function: sub_39B11B0
// Address: 0x39b11b0
//
_QWORD *__fastcall sub_39B11B0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rbx
  _QWORD *v6; // rax
  __int64 v8[8]; // [rsp+0h] [rbp-40h] BYREF

  sub_39B0CD0(v8, a2, a3);
  v3 = v8[0];
  v4 = v8[1];
  v5 = v8[2];
  v6 = (_QWORD *)sub_22077B0(0x20u);
  if ( v6 )
  {
    v6[1] = v3;
    v6[2] = v4;
    v6[3] = v5;
    *v6 = &unk_4A3FFC8;
  }
  *a1 = v6;
  return a1;
}
