// Function: sub_1EA9BD0
// Address: 0x1ea9bd0
//
__int64 (*__fastcall sub_1EA9BD0(__int64 a1, __int64 a2))(void)
{
  __int64 (*result)(void); // rax
  __int64 v3; // rdi

  sub_1F03410(a1, a2);
  result = *(__int64 (**)(void))(**(_QWORD **)(a1 + 2208) + 32LL);
  if ( (char *)result != (char *)nullsub_678 )
    result = (__int64 (*)(void))result();
  v3 = *(_QWORD *)(a1 + 2216);
  if ( v3 )
    return (__int64 (*)(void))(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v3 + 16LL))(v3, a2);
  return result;
}
