// Function: sub_107C6A0
// Address: 0x107c6a0
//
__int64 __fastcall sub_107C6A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v7; // r12
  __int64 v8; // rdi
  const void *v9; // r13
  __int64 result; // rax

  v7 = *(_QWORD *)(a2 + 8);
  v8 = a1[1];
  v9 = *(const void **)a2;
  if ( v8 + v7 > a1[2] )
  {
    result = sub_C8D290((__int64)a1, a1 + 3, v8 + v7, 1u, a5, a6);
    v8 = a1[1];
  }
  if ( v7 )
  {
    result = (__int64)memcpy((void *)(*a1 + v8), v9, v7);
    v8 = a1[1];
  }
  a1[1] = v8 + v7;
  return result;
}
