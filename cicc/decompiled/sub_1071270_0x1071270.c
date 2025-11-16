// Function: sub_1071270
// Address: 0x1071270
//
__int64 __fastcall sub_1071270(__int64 a1, unsigned __int8 *a2, size_t a3, int a4)
{
  __int64 v6; // r13
  void *v8; // rdi

  v6 = *(_QWORD *)(a1 + 2048);
  v8 = *(void **)(v6 + 32);
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v8 < a3 )
  {
    sub_CB6200(v6, a2, a3);
    v6 = *(_QWORD *)(a1 + 2048);
  }
  else if ( a3 )
  {
    memcpy(v8, a2, a3);
    *(_QWORD *)(v6 + 32) += a3;
    v6 = *(_QWORD *)(a1 + 2048);
  }
  return sub_CB6C70(v6, a4 - (int)a3);
}
