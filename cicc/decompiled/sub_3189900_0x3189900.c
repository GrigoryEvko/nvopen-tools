// Function: sub_3189900
// Address: 0x3189900
//
_QWORD *__fastcall sub_3189900(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  _QWORD *v6; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_22077B0(0x20u);
  v3 = (_QWORD *)v2;
  if ( v2 )
  {
    sub_318EB10(v2, 4, a2, a1);
    *v3 = &unk_4A32ED0;
    sub_371B4B0(v3, a2);
  }
  v6 = v3;
  v4 = sub_3189570(a1, (__int64)&v6);
  if ( v6 )
    (*(void (__fastcall **)(_QWORD *))(*v6 + 8LL))(v6);
  sub_371B4B0(v4, a2);
  return v4;
}
