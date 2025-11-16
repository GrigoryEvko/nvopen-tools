// Function: sub_1549FF0
// Address: 0x1549ff0
//
__int64 __fastcall sub_1549FF0(__int64 a1, const char *a2, size_t a3)
{
  void *v4; // rdi

  v4 = *(void **)(a1 + 24);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v4 < a3 )
    return sub_16E7EE0(a1, a2);
  if ( a3 )
  {
    memcpy(v4, a2, a3);
    *(_QWORD *)(a1 + 24) += a3;
  }
  return a1;
}
