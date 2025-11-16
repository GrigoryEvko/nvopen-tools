// Function: sub_2310AE0
// Address: 0x2310ae0
//
_BYTE *__fastcall sub_2310AE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v8; // rdi
  __int64 v9; // rax
  _WORD *v10; // rdx
  _BYTE *result; // rax

  v6 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v6) <= 6 )
  {
    v8 = sub_CB6200(a2, "devirt<", 7u);
  }
  else
  {
    *(_DWORD *)v6 = 1769366884;
    v8 = a2;
    *(_WORD *)(v6 + 4) = 29810;
    *(_BYTE *)(v6 + 6) = 60;
    *(_QWORD *)(a2 + 32) += 7LL;
  }
  v9 = sub_CB59F0(v8, *(int *)(a1 + 16));
  v10 = *(_WORD **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 1u )
  {
    sub_CB6200(v9, ">(", 2u);
  }
  else
  {
    *v10 = 10302;
    *(_QWORD *)(v9 + 32) += 2LL;
  }
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(
    *(_QWORD *)(a1 + 8),
    a2,
    a3,
    a4);
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 41);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 41;
  return result;
}
