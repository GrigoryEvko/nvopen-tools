// Function: sub_C82D40
// Address: 0xc82d40
//
__int64 __fastcall sub_C82D40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  __int64 result; // rax

  sub_C82C70(a1 + 16, a2, a3, a4, a5);
  v6 = *(_QWORD *)(a1 + 24);
  result = a1 + 40;
  if ( v6 != a1 + 40 )
    return j_j___libc_free_0(v6, *(_QWORD *)(a1 + 40) + 1LL);
  return result;
}
