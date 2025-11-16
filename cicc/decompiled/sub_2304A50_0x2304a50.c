// Function: sub_2304A50
// Address: 0x2304a50
//
_QWORD *__fastcall sub_2304A50(_QWORD *a1, __int64 a2, const char *a3, __int64 a4)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_FE8080(&v7, a2 + 8, a3, a4);
  sub_FDC100(v8, &v7);
  v4 = (_QWORD *)sub_22077B0(0x10u);
  v5 = v4;
  if ( v4 )
  {
    *v4 = &unk_4A0B150;
    sub_FDC100(v4 + 1, v8);
  }
  sub_FDC110(v8);
  *a1 = v5;
  sub_FDC110(&v7);
  return a1;
}
