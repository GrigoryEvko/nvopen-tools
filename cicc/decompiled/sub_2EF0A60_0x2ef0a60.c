// Function: sub_2EF0A60
// Address: 0x2ef0a60
//
_BYTE *__fastcall sub_2EF0A60(__int64 a1, char *a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v9; // rdi
  void *v10; // rdx
  __int64 v11; // rax
  _DWORD *v12; // rdx
  __int64 v13; // rdi
  _BYTE *result; // rax

  sub_2EF06E0(a1, a2, *(_QWORD *)(a3 + 16));
  v9 = *(_QWORD *)(a1 + 16);
  v10 = *(void **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 9u )
  {
    v9 = sub_CB6200(v9, "- operand ", 0xAu);
  }
  else
  {
    qmemcpy(v10, "- operand ", 10);
    *(_QWORD *)(v9 + 32) += 10LL;
  }
  v11 = sub_CB59D0(v9, a4);
  v12 = *(_DWORD **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 3u )
  {
    sub_CB6200(v11, (unsigned __int8 *)":   ", 4u);
  }
  else
  {
    *v12 = 538976314;
    *(_QWORD *)(v11 + 32) += 4LL;
  }
  sub_2EAF8F0(a3, *(_QWORD *)(a1 + 16), a5, *(_QWORD *)(a1 + 56));
  v13 = *(_QWORD *)(a1 + 16);
  result = *(_BYTE **)(v13 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v13 + 24) )
    return (_BYTE *)sub_CB5D20(v13, 10);
  *(_QWORD *)(v13 + 32) = result + 1;
  *result = 10;
  return result;
}
