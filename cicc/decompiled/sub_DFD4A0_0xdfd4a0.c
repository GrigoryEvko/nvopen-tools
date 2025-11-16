// Function: sub_DFD4A0
// Address: 0xdfd4a0
//
__int64 __fastcall sub_DFD4A0(__int64 *a1)
{
  __int64 v1; // rdi
  __int64 (*v2)(); // r10

  v1 = *a1;
  v2 = *(__int64 (**)())(*(_QWORD *)v1 + 1272LL);
  if ( v2 == sub_DF61A0 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64))v2)(v1);
}
