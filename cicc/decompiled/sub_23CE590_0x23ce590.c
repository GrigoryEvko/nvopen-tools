// Function: sub_23CE590
// Address: 0x23ce590
//
_QWORD *__fastcall sub_23CE590(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rsi
  _QWORD *(__fastcall *v4)(_QWORD *, __int64, __int64); // rax
  __int64 v5; // rax

  v3 = *a2;
  v4 = *(_QWORD *(__fastcall **)(_QWORD *, __int64, __int64))(*(_QWORD *)v3 + 104LL);
  if ( v4 == sub_23CE410 )
  {
    v5 = sub_B2BEC0(a3);
    sub_DF9330(a1, v5);
  }
  else
  {
    v4(a1, v3, a3);
  }
  return a1;
}
