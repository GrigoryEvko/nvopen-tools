// Function: sub_DFDC10
// Address: 0xdfdc10
//
__int64 __fastcall sub_DFDC10(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdi
  __int64 (*v5)(); // r9

  v4 = *a1;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 1328LL);
  if ( v5 == sub_DF6210 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64))v5)(v4, a2, a3, a4);
}
