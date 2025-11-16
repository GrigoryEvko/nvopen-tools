// Function: sub_21BE150
// Address: 0x21be150
//
__int64 __fastcall sub_21BE150(__int64 a1)
{
  __int64 v1; // rdi
  __int64 (__fastcall *v2)(__int64); // rax
  __int64 v4; // rax

  v1 = *(_QWORD *)(a1 + 480);
  v2 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v1 + 56LL);
  if ( v2 == sub_214ABA0 )
    return sub_21CF340(v1 + 696);
  v4 = ((__int64 (*)(void))v2)();
  return sub_21CF340(v4);
}
