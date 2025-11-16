// Function: sub_DFBC30
// Address: 0xdfbc30
//
__int64 __fastcall sub_DFBC30(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v11; // rdi
  __int64 (*v12)(); // r10

  v11 = *a1;
  v12 = *(__int64 (**)())(*(_QWORD *)v11 + 1192LL);
  if ( v12 == sub_DF6110 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64, _QWORD, __int64, __int64, __int64, __int64))v12)(
             v11,
             a2,
             a3,
             a4,
             a5,
             a6,
             a7,
             a8,
             a9,
             a10,
             a11);
}
