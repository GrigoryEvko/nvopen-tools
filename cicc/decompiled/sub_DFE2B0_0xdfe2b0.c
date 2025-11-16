// Function: sub_DFE2B0
// Address: 0xdfe2b0
//
__int64 __fastcall sub_DFE2B0(__int64 *a1, unsigned int a2)
{
  __int64 v2; // rdi
  __int64 result; // rax
  __int64 (__fastcall *v4)(__int64, unsigned int); // r9

  v2 = *a1;
  result = a2;
  v4 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v2 + 1536LL);
  if ( v4 != sub_DF62E0 )
    return v4(v2, a2);
  return result;
}
