// Function: sub_D9C9D0
// Address: 0xd9c9d0
//
__int64 __fastcall sub_D9C9D0(_QWORD *a1, const void *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  size_t v7; // r12
  void *v9; // rdi
  __int64 v10; // r13

  result = 0x400000000LL;
  v7 = 8 * a3;
  v9 = a1 + 2;
  *a1 = v9;
  v10 = (8 * a3) >> 3;
  a1[1] = 0x400000000LL;
  if ( (unsigned __int64)(8 * a3) > 0x20 )
  {
    sub_C8D5F0((__int64)a1, v9, (8 * a3) >> 3, 8u, a5, a6);
    v9 = (void *)(*a1 + 8LL * *((unsigned int *)a1 + 2));
  }
  else if ( !v7 )
  {
    *((_DWORD *)a1 + 2) = v10;
    return result;
  }
  result = (__int64)memcpy(v9, a2, v7);
  *((_DWORD *)a1 + 2) += v10;
  return result;
}
