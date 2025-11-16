// Function: sub_2D55440
// Address: 0x2d55440
//
void **__fastcall sub_2D55440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  void **result; // rax
  void **v9; // rsi
  __int64 v10; // rdi
  void **v11; // rdx

  v7 = *(unsigned int *)(a2 + 120);
  result = *(void ***)(a2 + 112);
  v9 = &result[v7];
  v10 = (8 * v7) >> 3;
  if ( (8 * v7) >> 5 )
  {
    v11 = &result[4 * ((8 * v7) >> 5)];
    while ( *result != &unk_4F8144C )
    {
      if ( result[1] == &unk_4F8144C )
      {
        ++result;
        break;
      }
      if ( result[2] == &unk_4F8144C )
      {
        result += 2;
        break;
      }
      if ( result[3] == &unk_4F8144C )
      {
        result += 3;
        break;
      }
      result += 4;
      if ( result == v11 )
      {
        v10 = v9 - result;
        goto LABEL_11;
      }
    }
LABEL_8:
    if ( v9 != result )
      return result;
    goto LABEL_15;
  }
LABEL_11:
  if ( v10 != 2 )
  {
    if ( v10 != 3 )
    {
      if ( v10 != 1 )
        goto LABEL_15;
      goto LABEL_14;
    }
    if ( *result == &unk_4F8144C )
      goto LABEL_8;
    ++result;
  }
  if ( *result == &unk_4F8144C )
    goto LABEL_8;
  ++result;
LABEL_14:
  if ( *result == &unk_4F8144C )
    goto LABEL_8;
LABEL_15:
  result = (void **)*(unsigned int *)(a2 + 124);
  if ( v7 + 1 > (unsigned __int64)result )
  {
    sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, a5, a6);
    result = *(void ***)(a2 + 112);
    v9 = &result[*(unsigned int *)(a2 + 120)];
  }
  *v9 = &unk_4F8144C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
