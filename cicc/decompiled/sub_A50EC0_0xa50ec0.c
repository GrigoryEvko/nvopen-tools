// Function: sub_A50EC0
// Address: 0xa50ec0
//
void __fastcall sub_A50EC0(__int64 a1, __int64 a2)
{
  const void *v2; // r14
  size_t v4; // rax
  void *v5; // rdi
  size_t v6; // r13

  if ( *(_BYTE *)a2 )
  {
    *(_BYTE *)a2 = 0;
  }
  else
  {
    v2 = *(const void **)(a2 + 8);
    if ( v2 )
    {
      v4 = strlen(*(const char **)(a2 + 8));
      v5 = *(void **)(a1 + 32);
      v6 = v4;
      if ( v4 > *(_QWORD *)(a1 + 24) - (_QWORD)v5 )
      {
        sub_CB6200(a1, v2, v4);
      }
      else if ( v4 )
      {
        memcpy(v5, v2, v4);
        *(_QWORD *)(a1 + 32) += v6;
      }
    }
  }
}
