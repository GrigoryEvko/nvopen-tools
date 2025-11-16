// Function: sub_15CBB30
// Address: 0x15cbb30
//
__int64 __fastcall sub_15CBB30(__int64 a1, const char *a2, size_t a3)
{
  __int64 v4; // r14
  void *v5; // rdi

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(void **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 < a3 )
  {
    sub_16E7EE0(v4, a2);
    return a1;
  }
  else
  {
    if ( a3 )
    {
      memcpy(v5, a2, a3);
      *(_QWORD *)(v4 + 24) += a3;
    }
    return a1;
  }
}
