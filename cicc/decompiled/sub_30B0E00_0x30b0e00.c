// Function: sub_30B0E00
// Address: 0x30b0e00
//
__int64 __fastcall sub_30B0E00(__int64 a1, int a2)
{
  char *v3; // r12
  size_t v4; // rax
  void *v5; // rdi
  size_t v6; // r14

  v3 = "memory";
  if ( a2 != 2 )
  {
    if ( a2 > 2 )
    {
      if ( a2 == 3 )
        v3 = "rooted";
    }
    else
    {
      v3 = "?? (error)";
      if ( a2 == 1 )
        v3 = "def-use";
    }
  }
  v4 = strlen(v3);
  v5 = *(void **)(a1 + 32);
  v6 = v4;
  if ( v4 > *(_QWORD *)(a1 + 24) - (_QWORD)v5 )
  {
    sub_CB6200(a1, (unsigned __int8 *)v3, v4);
    return a1;
  }
  else
  {
    if ( v4 )
    {
      memcpy(v5, v3, v4);
      *(_QWORD *)(a1 + 32) += v6;
    }
    return a1;
  }
}
