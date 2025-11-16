// Function: sub_BA97D0
// Address: 0xba97d0
//
_QWORD *__fastcall sub_BA97D0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12

  v2 = *(_QWORD *)(a2 + 160);
  if ( v2 )
  {
    *(_QWORD *)(a2 + 160) = 0;
    (*(void (__fastcall **)(_QWORD *, __int64))(*(_QWORD *)v2 + 24LL))(a1, v2);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  }
  else
  {
    *a1 = 1;
  }
  return a1;
}
