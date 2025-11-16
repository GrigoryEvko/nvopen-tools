// Function: sub_DFDCC0
// Address: 0xdfdcc0
//
__int64 __fastcall sub_DFDCC0(__int64 *a1, __int64 a2, unsigned __int8 a3, __int64 a4)
{
  __int64 v4; // rdi
  __int64 (*v5)(); // rax

  v4 = *a1;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 1344LL);
  if ( v5 == sub_DF6230 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64))v5)(v4, a2, a3, a4);
}
