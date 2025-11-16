// Function: sub_23FEE00
// Address: 0x23fee00
//
__int64 *__fastcall sub_23FEE00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  _QWORD *v6; // rdi
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *result; // rax
  __m128i v12; // [rsp+0h] [rbp-30h] BYREF
  char v13; // [rsp+18h] [rbp-18h]

  v3 = a1 + 32;
  v6 = (_QWORD *)(a1 + 96);
  *(v6 - 11) = v3;
  *(v6 - 8) = a2;
  *v6 = 0;
  v6[1] = 0;
  v6[2] = 0;
  *(v6 - 10) = 0x100000008LL;
  *((_DWORD *)v6 - 18) = 0;
  *((_BYTE *)v6 - 68) = 1;
  *(v6 - 12) = 1;
  v12.m128i_i64[0] = a2;
  v13 = 0;
  sub_23FEDC0((__int64)v6, &v12);
  if ( !*(_BYTE *)(a1 + 28) )
    return sub_C8CC70(a1, a3, (__int64)v7, v8, v9, v10);
  result = *(__int64 **)(a1 + 8);
  v8 = *(unsigned int *)(a1 + 20);
  v7 = &result[v8];
  if ( result == v7 )
  {
LABEL_7:
    if ( (unsigned int)v8 >= *(_DWORD *)(a1 + 16) )
      return sub_C8CC70(a1, a3, (__int64)v7, v8, v9, v10);
    *(_DWORD *)(a1 + 20) = v8 + 1;
    *v7 = a3;
    ++*(_QWORD *)a1;
  }
  else
  {
    while ( a3 != *result )
    {
      if ( v7 == ++result )
        goto LABEL_7;
    }
  }
  return result;
}
