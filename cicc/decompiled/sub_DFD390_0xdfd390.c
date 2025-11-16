// Function: sub_DFD390
// Address: 0xdfd390
//
__int64 __fastcall sub_DFD390(__int64 *a1)
{
  __int64 v1; // rdi
  __int64 (*v2)(); // r10

  v1 = *a1;
  v2 = *(__int64 (**)())(*(_QWORD *)v1 + 1240LL);
  if ( v2 == sub_DF6170 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64))v2)(v1);
}
