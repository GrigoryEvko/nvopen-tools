// Function: sub_2C6EAD0
// Address: 0x2c6ead0
//
__int64 __fastcall sub_2C6EAD0(__int64 a1, __int64 *a2)
{
  unsigned __int8 *v3; // rax
  size_t v4; // rdx
  void *v5; // rdi
  size_t v7; // [rsp+8h] [rbp-18h]

  v3 = (unsigned __int8 *)sub_BD5D20(*a2);
  v5 = *(void **)(a1 + 32);
  if ( v4 > *(_QWORD *)(a1 + 24) - (_QWORD)v5 )
    return sub_CB6200(a1, v3, v4);
  if ( v4 )
  {
    v7 = v4;
    memcpy(v5, v3, v4);
    *(_QWORD *)(a1 + 32) += v7;
  }
  return a1;
}
