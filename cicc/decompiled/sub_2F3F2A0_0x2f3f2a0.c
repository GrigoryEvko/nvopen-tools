// Function: sub_2F3F2A0
// Address: 0x2f3f2a0
//
unsigned __int64 __fastcall sub_2F3F2A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 result; // rax
  unsigned __int8 *v4; // r14
  size_t v5; // rax
  void *v6; // rdi
  size_t v7; // r13
  void *v8; // rdx

  v2 = a2;
  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result > 6 )
  {
    v8 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v8 <= 0xBu )
    {
      v2 = sub_CB6200(a2, "TargetCustom", 0xCu);
    }
    else
    {
      qmemcpy(v8, "TargetCustom", 12);
      *(_QWORD *)(a2 + 32) += 12LL;
    }
    return sub_CB59D0(v2, *(unsigned int *)(a1 + 8));
  }
  else
  {
    v4 = (unsigned __int8 *)*(&off_49D4420 + result);
    if ( v4 )
    {
      v5 = strlen((const char *)*(&off_49D4420 + result));
      v6 = *(void **)(a2 + 32);
      v7 = v5;
      result = *(_QWORD *)(a2 + 24) - (_QWORD)v6;
      if ( v7 > result )
      {
        return sub_CB6200(a2, v4, v7);
      }
      else if ( v7 )
      {
        result = (unsigned __int64)memcpy(v6, v4, v7);
        *(_QWORD *)(a2 + 32) += v7;
      }
    }
  }
  return result;
}
