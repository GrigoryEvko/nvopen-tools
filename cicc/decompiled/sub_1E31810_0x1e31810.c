// Function: sub_1E31810
// Address: 0x1e31810
//
_QWORD *__fastcall sub_1E31810(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdx
  _QWORD *result; // rax
  __int64 v7; // rdx
  char *v8; // r14
  size_t v9; // rax
  void *v10; // rdi
  size_t v11; // r13

  v5 = *(_QWORD **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v5 > 7u )
  {
    *v5 = 0x2E67657262757325LL;
    *(_QWORD *)(a1 + 24) += 8LL;
    if ( a3 )
      goto LABEL_3;
    return (_QWORD *)sub_16E7A90(a1, a2);
  }
  sub_16E7EE0(a1, "%subreg.", 8u);
  if ( !a3 )
    return (_QWORD *)sub_16E7A90(a1, a2);
LABEL_3:
  result = *(_QWORD **)(a3 + 240);
  v7 = (unsigned int)(a2 - 1);
  v8 = (char *)result[v7];
  if ( v8 )
  {
    v9 = strlen((const char *)result[v7]);
    v10 = *(void **)(a1 + 24);
    v11 = v9;
    result = (_QWORD *)(*(_QWORD *)(a1 + 16) - (_QWORD)v10);
    if ( v11 > (unsigned __int64)result )
    {
      return (_QWORD *)sub_16E7EE0(a1, v8, v11);
    }
    else if ( v11 )
    {
      result = memcpy(v10, v8, v11);
      *(_QWORD *)(a1 + 24) += v11;
    }
  }
  return result;
}
