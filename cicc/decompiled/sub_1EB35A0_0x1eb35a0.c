// Function: sub_1EB35A0
// Address: 0x1eb35a0
//
unsigned __int64 __fastcall sub_1EB35A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 result; // rax
  char *v4; // r14
  size_t v5; // rax
  void *v6; // rdi
  size_t v7; // r13
  void *v8; // rdx

  v2 = a2;
  result = *(int *)(a1 + 8);
  if ( (int)result > 6 )
  {
    v8 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v8 <= 0xBu )
    {
      v2 = sub_16E7EE0(a2, "TargetCustom", 0xCu);
    }
    else
    {
      qmemcpy(v8, "TargetCustom", 12);
      *(_QWORD *)(a2 + 24) += 12LL;
    }
    return sub_16E7AB0(v2, *(int *)(a1 + 8));
  }
  else
  {
    v4 = (char *)*(&off_49858E0 + result);
    if ( v4 )
    {
      v5 = strlen((const char *)*(&off_49858E0 + result));
      v6 = *(void **)(a2 + 24);
      v7 = v5;
      result = *(_QWORD *)(a2 + 16) - (_QWORD)v6;
      if ( v7 > result )
      {
        return sub_16E7EE0(a2, v4, v7);
      }
      else if ( v7 )
      {
        result = (unsigned __int64)memcpy(v6, v4, v7);
        *(_QWORD *)(a2 + 24) += v7;
      }
    }
  }
  return result;
}
