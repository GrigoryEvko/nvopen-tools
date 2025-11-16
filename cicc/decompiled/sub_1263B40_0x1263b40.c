// Function: sub_1263B40
// Address: 0x1263b40
//
__int64 __fastcall sub_1263B40(__int64 a1, const char *a2)
{
  size_t v3; // rax
  void *v4; // rdi
  size_t v5; // r14

  if ( !a2 )
    return a1;
  v3 = strlen(a2);
  v4 = *(void **)(a1 + 24);
  v5 = v3;
  if ( v3 <= *(_QWORD *)(a1 + 16) - (_QWORD)v4 )
  {
    if ( v3 )
    {
      memcpy(v4, a2, v3);
      *(_QWORD *)(a1 + 24) += v5;
    }
    return a1;
  }
  return sub_16E7EE0(a1, a2, v3);
}
