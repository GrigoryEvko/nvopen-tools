// Function: sub_DFE000
// Address: 0xdfe000
//
__int64 __fastcall sub_DFE000(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 (__fastcall *v6)(__int64, __int64, __int64, unsigned int); // r8

  v4 = *a1;
  result = a4;
  v6 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned int))(*(_QWORD *)v4 + 1448LL);
  if ( v6 != sub_DF6290 )
    return v6(v4, a2, a3, a4);
  return result;
}
