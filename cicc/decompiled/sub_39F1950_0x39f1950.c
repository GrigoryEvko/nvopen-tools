// Function: sub_39F1950
// Address: 0x39f1950
//
_BYTE *__fastcall sub_39F1950(unsigned int *a1, __int64 a2, __int64 a3, void *a4, size_t a5)
{
  __int64 v9; // rdx
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // r8
  char *v13; // rax
  size_t v14; // rdx
  void *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r14
  char *v18; // rdi
  void *v19; // rdi
  _BYTE *result; // rax
  __int64 v21; // [rsp+0h] [rbp-40h]
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]
  size_t v24; // [rsp+8h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v9) <= 8 )
  {
    v10 = sub_16E7EE0(a2, "<MCInst #", 9u);
  }
  else
  {
    *(_BYTE *)(v9 + 8) = 35;
    v10 = a2;
    *(_QWORD *)v9 = 0x2074736E49434D3CLL;
    *(_QWORD *)(a2 + 24) += 9LL;
  }
  sub_16E7A90(v10, *a1);
  if ( a3 )
  {
    v11 = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)v11 >= *(_QWORD *)(a2 + 16) )
    {
      v12 = sub_16E7DE0(a2, 32);
    }
    else
    {
      v12 = a2;
      *(_QWORD *)(a2 + 24) = v11 + 1;
      *v11 = 32;
    }
    v22 = v12;
    v13 = (char *)sub_38D0530(a3, *a1);
    v15 = *(void **)(v22 + 24);
    if ( v14 > *(_QWORD *)(v22 + 16) - (_QWORD)v15 )
    {
      sub_16E7EE0(v22, v13, v14);
    }
    else if ( v14 )
    {
      v21 = v22;
      v24 = v14;
      memcpy(v15, v13, v14);
      *(_QWORD *)(v21 + 24) += v24;
    }
  }
  v16 = a1[6];
  v23 = 16 * v16;
  v17 = 0;
  if ( (_DWORD)v16 )
  {
    do
    {
      v19 = *(void **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v19 >= a5 )
      {
        if ( a5 )
        {
          memcpy(v19, a4, a5);
          *(_QWORD *)(a2 + 24) += a5;
        }
      }
      else
      {
        sub_16E7EE0(a2, (char *)a4, a5);
      }
      v18 = (char *)(v17 + *((_QWORD *)a1 + 2));
      v17 += 16;
      sub_39F15E0(v18, a2);
    }
    while ( v17 != v23 );
  }
  result = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a2, ">", 1u);
  *result = 62;
  ++*(_QWORD *)(a2 + 24);
  return result;
}
