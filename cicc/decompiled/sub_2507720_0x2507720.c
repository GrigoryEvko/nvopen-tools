// Function: sub_2507720
// Address: 0x2507720
//
_BYTE *__fastcall sub_2507720(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r14
  void *v4; // rdx
  unsigned __int64 v5; // r12
  _BYTE *result; // rax

  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a1 + 16LL))(a1, 0, a2);
  v2 = *(_QWORD **)(a1 + 40);
  v3 = &v2[*(unsigned int *)(a1 + 48)];
  while ( v3 != v2 )
  {
    v4 = *(void **)(a2 + 32);
    v5 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 > 9u )
    {
      qmemcpy(v4, "  updates ", 10);
      *(_QWORD *)(a2 + 32) += 10LL;
    }
    else
    {
      sub_CB6200(a2, "  updates ", 0xAu);
    }
    ++v2;
    (*(void (__fastcall **)(unsigned __int64, _QWORD, __int64))(*(_QWORD *)v5 + 16LL))(v5, 0, a2);
  }
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 10);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 10;
  return result;
}
