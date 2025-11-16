// Function: sub_A58630
// Address: 0xa58630
//
_BYTE *__fastcall sub_A58630(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r14
  _BYTE *result; // rax
  __int64 v10; // r15
  unsigned int v11; // eax
  const void *v12; // rax
  size_t v13; // rdx
  void *v14; // rdi
  __int64 v15; // r12
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 v21; // r12
  size_t v22; // [rsp+0h] [rbp-60h]
  __int64 v23; // [rsp+8h] [rbp-58h]
  _QWORD v24[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v25[8]; // [rsp+20h] [rbp-40h] BYREF

  v7 = sub_A73280(a2, a2, a3, a4, a5, a6);
  v8 = sub_A73290(a2);
  result = a1 + 5;
  v23 = (__int64)(a1 + 5);
  if ( v7 != v8 )
  {
LABEL_2:
    if ( (unsigned __int8)sub_A71860(v7) )
    {
LABEL_3:
      v10 = *a1;
      v11 = sub_A71AE0(v7);
      v12 = (const void *)sub_A6FBB0(v11);
      v14 = *(void **)(v10 + 32);
      if ( *(_QWORD *)(v10 + 24) - (_QWORD)v14 < v13 )
      {
        sub_CB6200(v10, v12, v13);
      }
      else if ( v13 )
      {
        v22 = v13;
        memcpy(v14, v12, v13);
        *(_QWORD *)(v10 + 32) += v22;
      }
      result = (_BYTE *)sub_A72A60(v7);
      v15 = (__int64)result;
      if ( result )
      {
        v16 = *a1;
        v17 = *(_BYTE **)(*a1 + 32);
        if ( (unsigned __int64)v17 >= *(_QWORD *)(*a1 + 24) )
        {
          sub_CB5D20(v16, 40);
        }
        else
        {
          *(_QWORD *)(v16 + 32) = v17 + 1;
          *v17 = 40;
        }
        sub_A57EC0(v23, v15, *a1);
        v18 = *a1;
        result = *(_BYTE **)(*a1 + 32);
        if ( (unsigned __int64)result >= *(_QWORD *)(*a1 + 24) )
        {
          result = (_BYTE *)sub_CB5D20(v18, 41);
        }
        else
        {
          *(_QWORD *)(v18 + 32) = result + 1;
          *result = 41;
        }
      }
    }
    else
    {
      while ( 1 )
      {
        v21 = *a1;
        sub_A759D0(v24, v7, 0);
        sub_CB6200(v21, v24[0], v24[1]);
        result = v25;
        if ( (_QWORD *)v24[0] == v25 )
          break;
        v7 += 8;
        result = (_BYTE *)j_j___libc_free_0(v24[0], v25[0] + 1LL);
        if ( v8 == v7 )
          return result;
LABEL_12:
        v19 = *a1;
        v20 = *(_BYTE **)(*a1 + 32);
        if ( (unsigned __int64)v20 >= *(_QWORD *)(*a1 + 24) )
        {
          sub_CB5D20(v19, 32);
          goto LABEL_2;
        }
        *(_QWORD *)(v19 + 32) = v20 + 1;
        *v20 = 32;
        if ( (unsigned __int8)sub_A71860(v7) )
          goto LABEL_3;
      }
    }
    v7 += 8;
    if ( v8 != v7 )
      goto LABEL_12;
  }
  return result;
}
