// Function: sub_DFD610
// Address: 0xdfd610
//
__int64 __fastcall sub_DFD610(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdi
  __int64 (*v6)(); // r10

  v5 = *a1;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 1320LL);
  if ( v6 == sub_DF6200 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v6)(v5, a2, a3, a4, a5);
}
