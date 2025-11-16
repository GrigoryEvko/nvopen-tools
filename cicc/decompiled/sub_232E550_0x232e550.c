// Function: sub_232E550
// Address: 0x232e550
//
__int64 __fastcall sub_232E550(const void *a1, size_t a2, unsigned __int8 *a3, size_t a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // r12
  _BYTE *v9; // rax
  void *v10; // rdi
  __int64 v12; // rax

  v7 = sub_A51340(a5, "  ", 2u);
  v8 = sub_A51340(v7, a1, a2);
  v9 = *(_BYTE **)(v8 + 32);
  if ( *(_BYTE **)(v8 + 24) == v9 )
  {
    v12 = sub_CB6200(v8, "<", 1u);
    v10 = *(void **)(v12 + 32);
    v8 = v12;
  }
  else
  {
    *v9 = 60;
    v10 = (void *)(*(_QWORD *)(v8 + 32) + 1LL);
    *(_QWORD *)(v8 + 32) = v10;
  }
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v10 < a4 )
  {
    v8 = sub_CB6200(v8, a3, a4);
  }
  else if ( a4 )
  {
    memcpy(v10, a3, a4);
    *(_QWORD *)(v8 + 32) += a4;
  }
  return sub_A51340(v8, ">\n", 2u);
}
