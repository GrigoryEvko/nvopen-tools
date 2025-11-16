// Function: sub_39C1A80
// Address: 0x39c1a80
//
_QWORD *__fastcall sub_39C1A80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 *v9; // rcx
  _QWORD *result; // rax
  __m128i v11; // [rsp+0h] [rbp-20h] BYREF

  v11.m128i_i64[0] = a2;
  v11.m128i_i64[1] = a3;
  v7 = sub_39C1660(a1, &v11);
  v8 = *(unsigned int *)(v7 + 8);
  if ( (_DWORD)v8 )
  {
    v9 = (__int64 *)(*(_QWORD *)v7 + 16LL * (unsigned int)v8 - 16);
    if ( !v9[1] )
    {
      result = (_QWORD *)sub_1E15D60(*v9, a4, 0);
      if ( (_BYTE)result )
        return result;
      v8 = *(unsigned int *)(v7 + 8);
    }
  }
  if ( *(_DWORD *)(v7 + 12) <= (unsigned int)v8 )
  {
    sub_16CD150(v7, (const void *)(v7 + 16), 0, 16, v5, v6);
    v8 = *(unsigned int *)(v7 + 8);
  }
  result = (_QWORD *)(*(_QWORD *)v7 + 16 * v8);
  *result = a4;
  result[1] = 0;
  ++*(_DWORD *)(v7 + 8);
  return result;
}
