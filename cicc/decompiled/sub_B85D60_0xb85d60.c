// Function: sub_B85D60
// Address: 0xb85d60
//
unsigned __int64 __fastcall sub_B85D60(__int64 a1)
{
  _QWORD *v1; // rbx
  unsigned __int64 result; // rax
  _QWORD *i; // r13
  __int64 v5; // r15
  __int64 v6; // rdi
  __int64 v7; // rdi
  unsigned __int64 v8; // r15
  __int64 v9; // rax
  _WORD *v10; // rdx
  __int64 v11; // r8
  void *v12; // rdi
  const void *v13; // rsi
  size_t v14; // r15
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD **)(a1 + 16);
  result = *(unsigned int *)(a1 + 24);
  for ( i = &v1[result]; i != v1; ++v1 )
  {
    v5 = *v1;
    v6 = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v1 + 120LL))(*v1);
    if ( v6 )
    {
      result = sub_B85D60(v6);
    }
    else
    {
      v7 = *(_QWORD *)(a1 + 8);
      result = sub_B85AD0(v7, *(_QWORD *)(v5 + 16));
      v8 = result;
      if ( result )
      {
        v9 = sub_C5F790(v7);
        v10 = *(_WORD **)(v9 + 32);
        v11 = v9;
        if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 1u )
        {
          v15 = sub_CB6200(v9, " -", 2);
          v12 = *(void **)(v15 + 32);
          v11 = v15;
        }
        else
        {
          *v10 = 11552;
          v12 = (void *)(*(_QWORD *)(v9 + 32) + 2LL);
          *(_QWORD *)(v9 + 32) = v12;
        }
        v13 = *(const void **)(v8 + 16);
        v14 = *(_QWORD *)(v8 + 24);
        result = *(_QWORD *)(v11 + 24) - (_QWORD)v12;
        if ( v14 > result )
        {
          result = sub_CB6200(v11, v13, v14);
        }
        else if ( v14 )
        {
          v16 = v11;
          result = (unsigned __int64)memcpy(v12, v13, v14);
          *(_QWORD *)(v16 + 32) += v14;
        }
      }
    }
  }
  return result;
}
