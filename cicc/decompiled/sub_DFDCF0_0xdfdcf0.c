// Function: sub_DFDCF0
// Address: 0xdfdcf0
//
__int64 __fastcall sub_DFDCF0(__int64 *a1, unsigned __int8 a2)
{
  __int64 v2; // rdi
  __int64 (*v3)(); // rax

  v2 = *a1;
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 1352LL);
  if ( v3 == sub_DF6240 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64, _QWORD))v3)(v2, a2);
}
