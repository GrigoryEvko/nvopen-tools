// Function: sub_23AEE80
// Address: 0x23aee80
//
_QWORD *__fastcall sub_23AEE80(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  _QWORD *v6; // rbx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9

  v2 = (_QWORD *)sub_22077B0(0x68u);
  v6 = v2;
  if ( v2 )
  {
    *v2 = &unk_4A16218;
    sub_C8CD80((__int64)(v2 + 1), (__int64)(v2 + 5), a2 + 8, v3, v4, v5);
    sub_C8CD80((__int64)(v6 + 7), (__int64)(v6 + 11), a2 + 56, v7, v8, v9);
  }
  *a1 = v6;
  return a1;
}
