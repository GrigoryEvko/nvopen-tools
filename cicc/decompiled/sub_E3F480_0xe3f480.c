// Function: sub_E3F480
// Address: 0xe3f480
//
__int64 __fastcall sub_E3F480(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  __int64 v4; // r14
  void *v5; // rdi

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(void **)(v4 + 32);
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v5 < a3 )
  {
    sub_CB6200(v4, a2, a3);
    return a1;
  }
  else
  {
    if ( a3 )
    {
      memcpy(v5, a2, a3);
      *(_QWORD *)(v4 + 32) += a3;
    }
    return a1;
  }
}
