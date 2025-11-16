// Function: sub_ED6B40
// Address: 0xed6b40
//
void __fastcall sub_ED6B40(__int64 a1, char *a2)
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
      sub_CB6200(a1, (unsigned __int8 *)a2, v3);
    }
    else if ( v3 )
    {
      memcpy(v4, a2, v3);
      *(_QWORD *)(a1 + 32) += v5;
    }
  }
}
