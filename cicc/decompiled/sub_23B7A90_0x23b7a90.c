// Function: sub_23B7A90
// Address: 0x23b7a90
//
void __fastcall sub_23B7A90(
        __int64 a1,
        unsigned __int8 *a2,
        size_t a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        _QWORD *a13)
{
  __int64 v15; // rax
  void *v16; // rdi
  __int64 v17; // r13
  __int64 v18; // rax
  _QWORD v19[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a6 )
  {
    v15 = sub_904010(*(_QWORD *)(a1 + 40), "\n*** IR for function ");
    v16 = *(void **)(v15 + 32);
    v17 = v15;
    if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 < a3 )
    {
      v18 = sub_CB6200(v15, a2, a3);
      sub_904010(v18, " ***\n");
    }
    else
    {
      if ( a3 )
      {
        memcpy(v16, a2, a3);
        *(_QWORD *)(v17 + 32) += a3;
      }
      sub_904010(v17, " ***\n");
    }
  }
  v19[0] = a1;
  sub_23B7450(a12, a13, sub_23B08D0, (__int64)v19);
}
