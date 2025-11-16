// Function: sub_2F39590
// Address: 0x2f39590
//
__int64 (*__fastcall sub_2F39590(__int64 a1, __int64 a2))(void)
{
  __int64 (*result)(void); // rax
  __int64 v3; // rdi

  sub_2F90C60();
  result = *(__int64 (**)(void))(**(_QWORD **)(a1 + 3560) + 32LL);
  if ( (char *)result != (char *)nullsub_1618 )
    result = (__int64 (*)(void))result();
  v3 = *(_QWORD *)(a1 + 3568);
  if ( v3 )
    return (__int64 (*)(void))(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v3 + 16LL))(v3, a2);
  return result;
}
