// Function: sub_39B9370
// Address: 0x39b9370
//
__int64 __fastcall sub_39B9370(_BYTE **a1, _BYTE **a2)
{
  __int64 result; // rax
  size_t v3; // rbx
  size_t v4; // r12
  size_t *v5; // rdi
  const void *v6; // rdi
  size_t *v7; // rsi
  const void *v8; // rsi
  int v9; // eax
  int v10; // eax

  if ( (**a1 & 4) == 0 )
  {
    result = 0;
    if ( (**a2 & 4) == 0 )
      return result;
    v3 = 0;
    v4 = **((_QWORD **)*a2 - 1);
    goto LABEL_4;
  }
  v5 = (size_t *)*((_QWORD *)*a1 - 1);
  v3 = *v5;
  v6 = v5 + 2;
  if ( (**a2 & 4) == 0 )
    return v3 != 0;
  v7 = (size_t *)*((_QWORD *)*a2 - 1);
  v4 = *v7;
  v8 = v7 + 2;
  if ( v3 <= v4 )
  {
    if ( v3 )
    {
      v10 = memcmp(v6, v8, v3);
      if ( v10 )
        return (v10 >> 31) | 1u;
    }
LABEL_4:
    if ( v3 == v4 )
      return 0;
    return v3 < v4 ? -1 : 1;
  }
  if ( !v4 )
    return 1;
  v9 = memcmp(v6, v8, v4);
  if ( v9 )
    return (v9 >> 31) | 1u;
  return v3 < v4 ? -1 : 1;
}
