// Function: sub_16155F0
// Address: 0x16155f0
//
unsigned __int64 __fastcall sub_16155F0(__int64 a1)
{
  _QWORD *v1; // rbx
  unsigned __int64 result; // rax
  _QWORD *i; // r13
  __int64 v5; // r15
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rdx
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  _WORD *v12; // rdx
  __int64 v13; // r8
  void *v14; // rdi
  const char *v15; // rsi
  size_t v16; // r15
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD **)(a1 + 24);
  result = *(unsigned int *)(a1 + 32);
  for ( i = &v1[result]; i != v1; ++v1 )
  {
    v5 = *v1;
    v6 = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v1 + 120LL))(*v1);
    if ( v6 )
    {
      result = sub_16155F0(v6);
    }
    else
    {
      v7 = *(_QWORD *)(v5 + 16);
      v8 = *(_QWORD *)(a1 + 16);
      result = sub_1614F20(v8, v7);
      v10 = result;
      if ( result && !*(_BYTE *)(result + 42) )
      {
        v11 = sub_16BA580(v8, v7, v9);
        v12 = *(_WORD **)(v11 + 24);
        v13 = v11;
        if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 1u )
        {
          v17 = sub_16E7EE0(v11, " -", 2);
          v14 = *(void **)(v17 + 24);
          v13 = v17;
        }
        else
        {
          *v12 = 11552;
          v14 = (void *)(*(_QWORD *)(v11 + 24) + 2LL);
          *(_QWORD *)(v11 + 24) = v14;
        }
        v15 = *(const char **)(v10 + 16);
        v16 = *(_QWORD *)(v10 + 24);
        result = *(_QWORD *)(v13 + 16) - (_QWORD)v14;
        if ( v16 > result )
        {
          result = sub_16E7EE0(v13, v15, v16);
        }
        else if ( v16 )
        {
          v18 = v13;
          result = (unsigned __int64)memcpy(v14, v15, v16);
          *(_QWORD *)(v18 + 24) += v16;
        }
      }
    }
  }
  return result;
}
