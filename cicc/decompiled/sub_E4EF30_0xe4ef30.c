// Function: sub_E4EF30
// Address: 0xe4ef30
//
_BYTE *__fastcall sub_E4EF30(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  __int64 v4; // r13
  size_t v5; // r14
  void *v6; // rdi
  size_t v7; // rax

  if ( a3 )
  {
    v4 = *(_QWORD *)(a1 + 304);
    v5 = a3 - 1;
    v6 = *(void **)(v4 + 32);
    v7 = *(_QWORD *)(v4 + 24) - (_QWORD)v6;
    if ( a2[a3 - 1] == 10 )
    {
      if ( v5 > v7 )
      {
LABEL_4:
        sub_CB6200(v4, a2, v5);
        return sub_E4D880(a1);
      }
      if ( a3 == 1 )
        return sub_E4D880(a1);
    }
    else
    {
      v5 = a3;
      if ( v7 < a3 )
        goto LABEL_4;
    }
    memcpy(v6, a2, v5);
    *(_QWORD *)(v4 + 32) += v5;
  }
  return sub_E4D880(a1);
}
