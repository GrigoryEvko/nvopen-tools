// Function: sub_16D93B0
// Address: 0x16d93b0
//
__int64 __fastcall sub_16D93B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 result; // rax

  v7 = *(_QWORD **)(a1 + 136);
  if ( v7 )
    sub_16D9260(v7, (__int64 (*)())a1, a3, a4, a5, a6);
  v8 = *(_QWORD *)(a1 + 96);
  if ( v8 != a1 + 112 )
    j_j___libc_free_0(v8, *(_QWORD *)(a1 + 112) + 1LL);
  v9 = *(_QWORD *)(a1 + 64);
  result = a1 + 80;
  if ( v9 != a1 + 80 )
    return j_j___libc_free_0(v9, *(_QWORD *)(a1 + 80) + 1LL);
  return result;
}
