// Function: sub_B52E10
// Address: 0xb52e10
//
__int64 __fastcall sub_B52E10(__int64 a1, int a2)
{
  char *v3; // rax
  size_t v4; // rdx
  void *v5; // rdi
  size_t v7; // [rsp+8h] [rbp-18h]

  v3 = sub_B52C80(a2);
  v5 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 < v4 )
  {
    sub_CB6200(a1, v3, v4);
    return a1;
  }
  else
  {
    if ( v4 )
    {
      v7 = v4;
      memcpy(v5, v3, v4);
      *(_QWORD *)(a1 + 32) += v7;
    }
    return a1;
  }
}
