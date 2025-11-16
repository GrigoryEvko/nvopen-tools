// Function: sub_DFB6F0
// Address: 0xdfb6f0
//
__int64 __fastcall sub_DFB6F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdi
  __int64 (*v7)(); // rax

  v6 = *a1;
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 1160LL);
  if ( v7 == sub_DF60E0 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64))v7)(v6, a2, a3, a4, a5, a6);
}
