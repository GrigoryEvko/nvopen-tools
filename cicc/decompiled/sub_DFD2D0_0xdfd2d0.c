// Function: sub_DFD2D0
// Address: 0xdfd2d0
//
__int64 __fastcall sub_DFD2D0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // r10

  v3 = *a1;
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 1224LL);
  if ( v4 == sub_DF6150 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64))v4)(v3, a2, a3);
}
