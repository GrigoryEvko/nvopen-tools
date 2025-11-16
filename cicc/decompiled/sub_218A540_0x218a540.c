// Function: sub_218A540
// Address: 0x218a540
//
_BYTE *__fastcall sub_218A540(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  _BYTE *result; // rax
  void *v5; // rdi
  size_t *v6; // rsi
  size_t v8; // r13
  void *v9; // rsi

  result = *(_BYTE **)(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) + 24LL);
  if ( (*result & 4) != 0 )
  {
    v5 = *(void **)(a4 + 24);
    v6 = (size_t *)*((_QWORD *)result - 1);
    v8 = *v6;
    v9 = v6 + 2;
    result = (_BYTE *)(*(_QWORD *)(a4 + 16) - (_QWORD)v5);
    if ( (unsigned __int64)result >= v8 )
    {
      if ( v8 )
      {
        result = memcpy(v5, v9, v8);
        *(_QWORD *)(a4 + 24) += v8;
      }
    }
    else
    {
      return (_BYTE *)sub_16E7EE0(a4, (char *)v9, v8);
    }
  }
  return result;
}
