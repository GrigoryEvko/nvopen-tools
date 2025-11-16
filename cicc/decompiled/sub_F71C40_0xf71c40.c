// Function: sub_F71C40
// Address: 0xf71c40
//
__int64 __fastcall sub_F71C40(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  void *v8; // rdi
  unsigned int v9; // r13d
  size_t v10; // rdx

  result = 0x400000000LL;
  v8 = a1 + 2;
  *a1 = v8;
  a1[1] = 0x400000000LL;
  v9 = *(_DWORD *)(a2 + 8);
  if ( v9 && a1 != (_QWORD *)a2 )
  {
    v10 = 8LL * v9;
    if ( v9 <= 4
      || (result = sub_C8D5F0((__int64)a1, v8, v9, 8u, v9, a6),
          v8 = (void *)*a1,
          (v10 = 8LL * *(unsigned int *)(a2 + 8)) != 0) )
    {
      result = (__int64)memcpy(v8, *(const void **)a2, v10);
    }
    *((_DWORD *)a1 + 2) = v9;
  }
  return result;
}
