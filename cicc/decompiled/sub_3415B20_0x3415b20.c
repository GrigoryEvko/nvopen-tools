// Function: sub_3415B20
// Address: 0x3415b20
//
void __fastcall sub_3415B20(__int64 a1, __int64 a2)
{
  _QWORD *i; // rbx
  void (*v3)(); // rax
  _QWORD *v4; // rax
  __int64 v5; // r14
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *j; // rbx

  if ( (unsigned __int8)sub_33C7D40(a2)
    || (v4 = sub_C65C80(
               (__int64 *)(a1 + 520),
               (__int64 *)a2,
               (void (__fastcall **)(__int64 *, __int64 *, _QWORD *))off_4A367D0),
        v5 = (__int64)v4,
        (_QWORD *)a2 == v4) )
  {
    for ( i = *(_QWORD **)(a1 + 768); i; i = (_QWORD *)i[1] )
    {
      while ( 1 )
      {
        v3 = *(void (**)())(*i + 24LL);
        if ( v3 != nullsub_1873 )
          break;
        i = (_QWORD *)i[1];
        if ( !i )
          return;
      }
      ((void (__fastcall *)(_QWORD *, __int64))v3)(i, a2);
    }
  }
  else
  {
    sub_33D00A0((__int64)v4, *(_DWORD *)(a2 + 28));
    sub_34158F0(a1, a2, v5, v6, v7, v8);
    for ( j = *(_QWORD **)(a1 + 768); j; j = (_QWORD *)j[1] )
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*j + 16LL))(j, a2, v5);
    sub_33CEC60(a1, a2);
  }
}
