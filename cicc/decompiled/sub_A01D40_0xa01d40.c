// Function: sub_A01D40
// Address: 0xa01d40
//
void __fastcall sub_A01D40(__int64 a1, const char *a2)
{
  size_t v3; // rax
  void *v4; // rdi
  size_t v5; // r14

  if ( a2 )
  {
    v3 = strlen(a2);
    v4 = *(void **)(a1 + 32);
    v5 = v3;
    if ( v3 > *(_QWORD *)(a1 + 24) - (_QWORD)v4 )
    {
      sub_CB6200(a1, a2, v3);
    }
    else if ( v3 )
    {
      memcpy(v4, a2, v3);
      *(_QWORD *)(a1 + 32) += v5;
    }
  }
}
