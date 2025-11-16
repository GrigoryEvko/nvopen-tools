// Function: sub_E2DAE0
// Address: 0xe2dae0
//
__int64 __fastcall sub_E2DAE0(__int64 a1, char **a2, unsigned int a3)
{
  __int64 *v5; // rdi
  unsigned __int64 (__fastcall *v6)(__int64, char **, unsigned int); // rax

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 24) + 24LL))(*(_QWORD *)(a1 + 24));
  sub_E2A040((__int64)a2);
  v5 = *(__int64 **)(a1 + 16);
  v6 = *(unsigned __int64 (__fastcall **)(__int64, char **, unsigned int))(*v5 + 16);
  if ( v6 == sub_E2CA10 )
    sub_E2C8E0(v5[2], a2, a3, 2u, "::");
  else
    v6((__int64)v5, a2, a3);
  return (*(__int64 (__fastcall **)(_QWORD, char **, _QWORD))(**(_QWORD **)(a1 + 24) + 32LL))(
           *(_QWORD *)(a1 + 24),
           a2,
           a3);
}
