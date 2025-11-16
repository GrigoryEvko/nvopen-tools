// Function: sub_3245010
// Address: 0x3245010
//
void __fastcall sub_3245010(_QWORD *a1, __int64 a2)
{
  _QWORD **v2; // r12
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 v5; // rsi
  _QWORD **v7; // rsi
  _QWORD *v8; // rdi

  v7 = (_QWORD **)*a1;
  v8 = a1 + 13;
  if ( v8[3] == v8[4] )
  {
    nullsub_2028();
  }
  else
  {
    v2 = v7;
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*v2[28] + 176LL))(v2[28], a2, 0);
    v3 = (__int64 *)v8[3];
    v4 = (__int64 *)v8[4];
    while ( v4 != v3 )
    {
      v5 = *v3++;
      sub_31F1760((__int64)v2, v5);
    }
    ((void (__fastcall *)(_QWORD **, _QWORD, const char *, _QWORD))(*v2)[53])(v2, 0, "EOM(3)", 0);
  }
}
