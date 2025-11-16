// Function: sub_3937EA0
// Address: 0x3937ea0
//
void __fastcall sub_3937EA0(__int64 a1, char *a2)
{
  size_t v3; // rax
  void *v4; // rdi
  size_t v5; // r14

  if ( a2 )
  {
    v3 = strlen(a2);
    v4 = *(void **)(a1 + 24);
    v5 = v3;
    if ( v3 > *(_QWORD *)(a1 + 16) - (_QWORD)v4 )
    {
      sub_16E7EE0(a1, a2, v3);
    }
    else if ( v3 )
    {
      memcpy(v4, a2, v3);
      *(_QWORD *)(a1 + 24) += v5;
    }
  }
}
