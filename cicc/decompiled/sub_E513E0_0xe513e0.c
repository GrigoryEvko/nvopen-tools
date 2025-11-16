// Function: sub_E513E0
// Address: 0xe513e0
//
_BYTE *__fastcall sub_E513E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rdi
  _BYTE *v6; // rax
  unsigned __int64 v7; // r13
  bool v8; // zf
  _BYTE *result; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  unsigned __int8 *v12; // rsi
  size_t v13; // rdx
  void *v14; // rdi

  v3 = *(_QWORD *)(a1 + 304);
  v4 = *(_QWORD *)(v3 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v4) <= 5 )
  {
    sub_CB6200(v3, "\t.def\t", 6u);
  }
  else
  {
    *(_DWORD *)v4 = 1701064201;
    *(_WORD *)(v4 + 4) = 2406;
    *(_QWORD *)(v3 + 32) += 6LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(_BYTE **)(v5 + 32);
  if ( (unsigned __int64)v6 >= *(_QWORD *)(v5 + 24) )
  {
    sub_CB5D20(v5, 59);
  }
  else
  {
    *(_QWORD *)(v5 + 32) = v6 + 1;
    *v6 = 59;
  }
  v7 = *(_QWORD *)(a1 + 344);
  if ( v7 )
  {
    v11 = *(_QWORD *)(a1 + 304);
    v12 = *(unsigned __int8 **)(a1 + 336);
    v13 = *(_QWORD *)(a1 + 344);
    v14 = *(void **)(v11 + 32);
    if ( v7 > *(_QWORD *)(v11 + 24) - (_QWORD)v14 )
    {
      sub_CB6200(*(_QWORD *)(a1 + 304), v12, v13);
    }
    else
    {
      memcpy(v14, v12, v13);
      *(_QWORD *)(v11 + 32) += v7;
    }
  }
  v8 = *(_BYTE *)(a1 + 745) == 0;
  *(_QWORD *)(a1 + 344) = 0;
  if ( !v8 )
    return (_BYTE *)sub_E4D630((__int64 *)a1);
  v10 = *(_QWORD *)(a1 + 304);
  result = *(_BYTE **)(v10 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v10 + 24) )
    return (_BYTE *)sub_CB5D20(v10, 10);
  *(_QWORD *)(v10 + 32) = result + 1;
  *result = 10;
  return result;
}
