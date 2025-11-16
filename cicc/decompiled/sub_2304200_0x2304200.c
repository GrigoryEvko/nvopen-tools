// Function: sub_2304200
// Address: 0x2304200
//
_QWORD *__fastcall sub_2304200(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r13
  _QWORD *v7; // rax

  v4 = sub_227B460(a2 + 8, a3, a4);
  v6 = v5;
  v7 = (_QWORD *)sub_22077B0(0x18u);
  if ( v7 )
  {
    v7[1] = v4;
    v7[2] = v6;
    *v7 = &unk_4A0ABD8;
  }
  *a1 = v7;
  return a1;
}
