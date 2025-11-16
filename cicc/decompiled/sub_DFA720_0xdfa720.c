// Function: sub_DFA720
// Address: 0xdfa720
//
__int64 __fastcall sub_DFA720(__int64 *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rax

  v3 = *a1;
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 648LL);
  if ( v4 == sub_DF5E60 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v4)(v3, a2, a3);
}
