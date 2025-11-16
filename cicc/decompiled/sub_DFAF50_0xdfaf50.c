// Function: sub_DFAF50
// Address: 0xdfaf50
//
__int64 __fastcall sub_DFAF50(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  __int64 v4; // r8
  __int64 (*v5)(); // rcx

  v3 = *a1;
  v4 = 1;
  v5 = *(__int64 (**)())(*(_QWORD *)v3 + 928LL);
  if ( v5 != sub_DF5F70 )
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 (*)(), __int64))v5)(v3, a2, a3, v5, 1);
  return v4;
}
