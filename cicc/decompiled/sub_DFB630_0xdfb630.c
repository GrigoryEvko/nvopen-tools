// Function: sub_DFB630
// Address: 0xdfb630
//
__int64 __fastcall sub_DFB630(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v5; // rdi
  __int64 (*v6)(); // rax

  v5 = *a1;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 1128LL);
  if ( v6 == sub_DF60D0 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD))v6)(v5, a2, a3, a4, a5);
}
