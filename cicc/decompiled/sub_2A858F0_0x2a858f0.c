// Function: sub_2A858F0
// Address: 0x2a858f0
//
void **__fastcall sub_2A858F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
    while ( &unk_4F8BE8C != *result )
    {
      if ( &unk_4F8BE8C == result[1] )
      {
        ++result;
        break;
      }
      if ( &unk_4F8BE8C == result[2] )
      {
        result += 2;
        break;
      }
      if ( &unk_4F8BE8C == result[3] )
      {
        result += 3;
        break;
      }
      result += 4;
      if ( v11 == result )
      {
        v10 = v9 - result;
        goto LABEL_11;
      }
    }
LABEL_8:
    if ( v9 != result )
      return result;
    goto LABEL_14;
  }
LABEL_11:
  if ( v10 != 2 )
  {
    if ( v10 != 3 )
    {
      if ( v10 != 1 )
        goto LABEL_14;
      goto LABEL_21;
    }
    if ( &unk_4F8BE8C == *result )
      goto LABEL_8;
    ++result;
  }
  if ( &unk_4F8BE8C == *result )
    goto LABEL_8;
  ++result;
LABEL_21:
  if ( &unk_4F8BE8C == *result )
    goto LABEL_8;
LABEL_14:
  result = (void **)*(unsigned int *)(a2 + 124);
  if ( v7 + 1 > (unsigned __int64)result )
  {
    sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, a5, a6);
    result = *(void ***)(a2 + 112);
    v9 = &result[*(unsigned int *)(a2 + 120)];
  }
  *v9 = &unk_4F8BE8C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
