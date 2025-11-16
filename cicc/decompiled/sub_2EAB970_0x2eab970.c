// Function: sub_2EAB970
// Address: 0x2eab970
//
_QWORD *__fastcall sub_2EAB970(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdx
  _QWORD *result; // rax
  __int64 v7; // rdx
  unsigned __int8 *v8; // r14
  size_t v9; // rax
  void *v10; // rdi
  size_t v11; // r13

  v5 = *(_QWORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 7u )
  {
    sub_CB6200(a1, "%subreg.", 8u);
  }
  else
  {
    *v5 = 0x2E67657262757325LL;
    *(_QWORD *)(a1 + 32) += 8LL;
  }
  if ( !a3 || !a2 || *(unsigned int *)(a3 + 96) <= a2 )
    return (_QWORD *)sub_CB59D0(a1, a2);
  result = *(_QWORD **)(a3 + 256);
  v7 = (unsigned int)(a2 - 1);
  v8 = (unsigned __int8 *)result[v7];
  if ( v8 )
  {
    v9 = strlen((const char *)result[v7]);
    v10 = *(void **)(a1 + 32);
    v11 = v9;
    result = (_QWORD *)(*(_QWORD *)(a1 + 24) - (_QWORD)v10);
    if ( v11 > (unsigned __int64)result )
    {
      return (_QWORD *)sub_CB6200(a1, v8, v11);
    }
    else if ( v11 )
    {
      result = memcpy(v10, v8, v11);
      *(_QWORD *)(a1 + 32) += v11;
    }
  }
  return result;
}
