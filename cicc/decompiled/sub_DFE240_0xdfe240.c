// Function: sub_DFE240
// Address: 0xdfe240
//
__int64 __fastcall sub_DFE240(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rax

  v3 = *a1;
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 1520LL);
  if ( v4 == sub_DF62D0 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64))v4)(v3, a2, a3);
}
