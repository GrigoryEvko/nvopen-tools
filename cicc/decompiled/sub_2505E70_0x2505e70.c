// Function: sub_2505E70
// Address: 0x2505e70
//
__int64 __fastcall sub_2505E70(__int64 **a1, __int64 a2)
{
  __int64 v2; // r9

  v2 = **a1;
  if ( v2 )
    return (*(unsigned int (__fastcall **)(__int64, __int64 *, _QWORD, __int64, _QWORD))(*(_QWORD *)v2 + 112LL))(
             v2,
             a1[1],
             *a1[2],
             a2,
             *a1[3])
         ^ 1;
  else
    return 0;
}
