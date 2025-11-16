// Function: sub_124B5D0
// Address: 0x124b5d0
//
void __fastcall sub_124B5D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  size_t v4; // r12
  unsigned __int8 *v5; // r14
  size_t v6; // rax
  void *v7; // rdi
  size_t v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = -1;
  if ( a4 && !sub_C93C90(a3, a4, 0xAu, v8) )
    v4 = v8[0];
  v5 = *(unsigned __int8 **)(a1 + 8);
  if ( v5 )
  {
    v6 = strlen(*(const char **)(a1 + 8));
    v7 = *(void **)(a2 + 32);
    if ( v6 <= v4 )
      v4 = v6;
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 < v4 )
    {
      sub_CB6200(a2, v5, v4);
    }
    else if ( v4 )
    {
      memcpy(v7, v5, v4);
      *(_QWORD *)(a2 + 32) += v4;
    }
  }
}
