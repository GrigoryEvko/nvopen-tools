// Function: sub_352FB00
// Address: 0x352fb00
//
void **__fastcall sub_352FB00(__int64 a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rcx
  void **result; // rax
  void **v7; // rsi
  __int64 v8; // rdi
  void **v9; // rdx

  sub_BB9660(a2, (__int64)&unk_50208C0);
  sub_BB9660(a2, (__int64)&unk_501EC08);
  sub_BB9660(a2, (__int64)&unk_4F87C64);
  v5 = *(unsigned int *)(a2 + 152);
  result = *(void ***)(a2 + 144);
  v7 = &result[v5];
  v8 = (8 * v5) >> 3;
  if ( (8 * v5) >> 5 )
  {
    v9 = &result[4 * ((8 * v5) >> 5)];
    while ( *result != &unk_501695C )
    {
      if ( result[1] == &unk_501695C )
      {
        ++result;
        break;
      }
      if ( result[2] == &unk_501695C )
      {
        result += 2;
        break;
      }
      if ( result[3] == &unk_501695C )
      {
        result += 3;
        break;
      }
      result += 4;
      if ( result == v9 )
      {
        v8 = v7 - result;
        goto LABEL_11;
      }
    }
LABEL_8:
    if ( v7 != result )
      return result;
    goto LABEL_15;
  }
LABEL_11:
  if ( v8 != 2 )
  {
    if ( v8 != 3 )
    {
      if ( v8 != 1 )
        goto LABEL_15;
      goto LABEL_14;
    }
    if ( *result == &unk_501695C )
      goto LABEL_8;
    ++result;
  }
  if ( *result == &unk_501695C )
    goto LABEL_8;
  ++result;
LABEL_14:
  if ( *result == &unk_501695C )
    goto LABEL_8;
LABEL_15:
  result = (void **)*(unsigned int *)(a2 + 156);
  if ( v5 + 1 > (unsigned __int64)result )
  {
    sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v5 + 1, 8u, v3, v4);
    result = *(void ***)(a2 + 144);
    v7 = &result[*(unsigned int *)(a2 + 152)];
  }
  *v7 = &unk_501695C;
  ++*(_DWORD *)(a2 + 152);
  return result;
}
