// Function: sub_E8DAA0
// Address: 0xe8daa0
//
__int64 __fastcall sub_E8DAA0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  char *v4; // r9
  __int64 (__fastcall *v5)(_QWORD *, __int64, unsigned int, __int64, __int64, char *); // rax

  sub_E98210(a1, a2);
  v5 = *(__int64 (__fastcall **)(_QWORD *, __int64, unsigned int, __int64, __int64, char *))*a1;
  if ( v5 == sub_E8D250 )
    return sub_E8CEC0(a1, a2, 0, v2, v3, v4);
  else
    return ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD))v5)(a1, a2, 0);
}
