// Function: sub_C1F8C0
// Address: 0xc1f8c0
//
int __fastcall sub_C1F8C0(__int64 a1, __int64 a2)
{
  size_t v2; // r12
  size_t v3; // rbx
  const void *v4; // rsi
  const void *v5; // rdi
  size_t v6; // rdx
  int result; // eax

  v2 = *(_QWORD *)(a2 + 8);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(const void **)a2;
  v5 = *(const void **)a1;
  if ( v3 > v2 )
  {
    if ( v4 == v5 )
      return v3 < v2 ? -1 : 1;
    v6 = v2;
  }
  else
  {
    if ( v4 == v5 )
      goto LABEL_7;
    v6 = v3;
  }
  if ( !v5 )
    return -1;
  if ( !v4 )
    return 1;
  result = memcmp(v5, v4, v6);
  if ( !result )
  {
LABEL_7:
    result = 0;
    if ( v3 == v2 )
      return result;
    return v3 < v2 ? -1 : 1;
  }
  return result;
}
