// Function: sub_A54F00
// Address: 0xa54f00
//
_BYTE *__fastcall sub_A54F00(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  int v4; // edi
  unsigned __int8 *v5; // r13
  char v6; // r15
  unsigned __int8 *v7; // rbx
  void *v8; // rdi
  _BYTE *result; // rax
  _BYTE *v10; // rax

  v4 = *a2;
  if ( (unsigned int)(v4 - 48) > 9 )
  {
    v5 = &a2[a3];
    if ( a2 == &a2[a3] )
    {
LABEL_9:
      v8 = *(void **)(a1 + 32);
      result = (_BYTE *)(*(_QWORD *)(a1 + 24) - (_QWORD)v8);
      if ( (unsigned __int64)result < a3 )
        return (_BYTE *)sub_CB6200(a1, a2, a3);
      if ( a3 )
      {
        result = memcpy(v8, a2, a3);
        *(_QWORD *)(a1 + 32) += a3;
      }
      return result;
    }
    v6 = *a2;
    v7 = a2 + 1;
    while ( isalnum(v4) || (unsigned __int8)(v6 - 45) <= 1u || v6 == 95 )
    {
      if ( v5 == v7 )
        goto LABEL_9;
      v4 = *v7++;
      v6 = v4;
    }
  }
  v10 = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(a1 + 24) )
  {
    sub_CB5D20(a1, 34);
  }
  else
  {
    *(_QWORD *)(a1 + 32) = v10 + 1;
    *v10 = 34;
  }
  sub_C92400(a2, a3, a1);
  result = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a1 + 24) )
    return (_BYTE *)sub_CB5D20(a1, 34);
  *(_QWORD *)(a1 + 32) = result + 1;
  *result = 34;
  return result;
}
