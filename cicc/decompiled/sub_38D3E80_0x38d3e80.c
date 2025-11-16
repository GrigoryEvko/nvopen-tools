// Function: sub_38D3E80
// Address: 0x38d3e80
//
__int64 (*__fastcall sub_38D3E80(__int64 a1))(void)
{
  __int64 (*result)(void); // rax

  result = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a1 + 264) + 24LL) + 56LL);
  if ( (char *)result != (char *)nullsub_1932 )
    return (__int64 (*)(void))result();
  return result;
}
