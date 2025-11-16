// Function: sub_35EFDA0
// Address: 0x35efda0
//
_QWORD *__fastcall sub_35EFDA0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  _QWORD *result; // rax
  void *v5; // rdi
  size_t *v6; // rsi
  size_t v8; // r13
  void *v9; // rsi

  result = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) + 16LL);
  if ( (result[1] & 1) != 0 )
  {
    v5 = *(void **)(a4 + 32);
    v6 = (size_t *)*(result - 1);
    v8 = *v6;
    v9 = v6 + 3;
    result = (_QWORD *)(*(_QWORD *)(a4 + 24) - (_QWORD)v5);
    if ( (unsigned __int64)result >= v8 )
    {
      if ( v8 )
      {
        result = memcpy(v5, v9, v8);
        *(_QWORD *)(a4 + 32) += v8;
      }
    }
    else
    {
      return (_QWORD *)sub_CB6200(a4, (unsigned __int8 *)v9, v8);
    }
  }
  return result;
}
