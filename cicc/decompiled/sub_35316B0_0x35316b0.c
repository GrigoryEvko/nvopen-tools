// Function: sub_35316B0
// Address: 0x35316b0
//
__int64 __fastcall sub_35316B0(const void *a1, const void *a2)
{
  size_t *v2; // rdi
  size_t v3; // r12
  const void *v4; // rdi
  size_t v5; // rbx
  __int64 result; // rax
  size_t *v7; // rsi
  size_t v8; // rdx
  const void *v9; // rsi
  int v10; // eax

  if ( (*(_BYTE *)(*(_QWORD *)a1 + 8LL) & 1) != 0 )
  {
    v2 = *(size_t **)(*(_QWORD *)a1 - 8LL);
    v3 = *v2;
    v4 = v2 + 3;
  }
  else
  {
    v3 = 0;
    v4 = 0;
  }
  if ( (*(_BYTE *)(*(_QWORD *)a2 + 8LL) & 1) != 0 )
  {
    v7 = *(size_t **)(*(_QWORD *)a2 - 8LL);
    v8 = v3;
    v5 = *v7;
    v9 = v7 + 3;
    if ( v5 <= v3 )
      v8 = v5;
    if ( v8 )
    {
      v10 = memcmp(v4, v9, v8);
      if ( v10 )
        return (v10 >> 31) | 1u;
    }
  }
  else
  {
    v5 = 0;
  }
  result = 0;
  if ( v5 != v3 )
    return v3 < v5 ? -1 : 1;
  return result;
}
