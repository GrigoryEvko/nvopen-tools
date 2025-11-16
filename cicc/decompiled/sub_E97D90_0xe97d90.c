// Function: sub_E97D90
// Address: 0xe97d90
//
unsigned __int64 __fastcall sub_E97D90(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r15
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int64 result; // rax
  void (*v7)(); // rdx

  v3 = (_QWORD *)a1[1];
  v4 = sub_E808D0(a3, 0, v3, 0);
  v5 = sub_E808D0(a2, 0, (_QWORD *)a1[1], 0);
  result = sub_E81A00(18, v5, v4, v3, 0);
  v7 = *(void (**)())(*a1 + 568LL);
  if ( v7 != nullsub_340 )
    return ((__int64 (__fastcall *)(_QWORD *, unsigned __int64))v7)(a1, result);
  return result;
}
