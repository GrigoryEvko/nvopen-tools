// Function: sub_B93110
// Address: 0xb93110
//
__int64 __fastcall sub_B93110(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  unsigned __int64 v6; // r12

  result = *(_QWORD *)(a1 + 8);
  if ( (result & 4) != 0 )
  {
    v6 = result & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(result & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFFBLL;
    sub_B92F50((const __m128i *)(result & 0xFFFFFFFFFFFFFFF8LL), 1, a3, a4, a5);
    if ( (*(_BYTE *)(v6 + 24) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(v6 + 32), 24LL * *(unsigned int *)(v6 + 40), 8);
    return j_j___libc_free_0(v6, 128);
  }
  return result;
}
