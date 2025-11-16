// Function: sub_22AA2C0
// Address: 0x22aa2c0
//
__int64 *__fastcall sub_22AA2C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rax
  _WORD *v9; // rdx
  __int64 v10; // r12
  __int64 *v11; // rax
  _BYTE *v12; // rax
  _QWORD *v13; // rdx
  __int64 *result; // rax
  __int64 *v15; // r12
  __int64 *v16; // rbx
  void *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 v21; // [rsp+0h] [rbp-40h]

  v6 = 0;
  v21 = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)v21 )
  {
    do
    {
      while ( 1 )
      {
        v13 = *(_QWORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v13 > 7u )
        {
          v7 = a2;
          *v13 = 0x20676E69646E6942LL;
          *(_QWORD *)(a2 + 32) += 8LL;
        }
        else
        {
          v7 = sub_CB6200(a2, "Binding ", 8u);
        }
        v8 = sub_CB59D0(v7, v6);
        v9 = *(_WORD **)(v8 + 32);
        if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 1u )
        {
          sub_CB6200(v8, (unsigned __int8 *)":\n", 2u);
        }
        else
        {
          *v9 = 2618;
          *(_QWORD *)(v8 + 32) += 2LL;
        }
        v10 = *(_QWORD *)a1 + 32 * v6;
        v11 = sub_22A9FC0(a3, *(_QWORD *)(v10 + 16));
        sub_22A94B0((unsigned __int8 **)v10, a2, (__int64)v11, a4);
        v12 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v12 )
          break;
        *v12 = 10;
        ++v6;
        ++*(_QWORD *)(a2 + 32);
        if ( v21 == v6 )
          goto LABEL_11;
      }
      ++v6;
      sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
    }
    while ( v21 != v6 );
  }
LABEL_11:
  result = (__int64 *)a1;
  if ( *(_DWORD *)(a1 + 64) )
  {
    result = *(__int64 **)(a1 + 56);
    v15 = &result[2 * *(unsigned int *)(a1 + 72)];
    if ( result != v15 )
    {
      while ( 1 )
      {
        v16 = result;
        if ( *result != -8192 && *result != -4096 )
          break;
        result += 2;
        if ( v15 == result )
          return result;
      }
      if ( v15 != result )
      {
        do
        {
          v17 = *(void **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v17 <= 0xDu )
          {
            v18 = sub_CB6200(a2, "Call bound to ", 0xEu);
          }
          else
          {
            qmemcpy(v17, "Call bound to ", 14);
            v18 = a2;
            *(_QWORD *)(a2 + 32) += 14LL;
          }
          v19 = sub_CB59D0(v18, *((unsigned int *)v16 + 2));
          v20 = *(_BYTE **)(v19 + 32);
          if ( *(_BYTE **)(v19 + 24) == v20 )
          {
            sub_CB6200(v19, (unsigned __int8 *)":", 1u);
          }
          else
          {
            *v20 = 58;
            ++*(_QWORD *)(v19 + 32);
          }
          sub_A69870(*v16, (_BYTE *)a2, 0);
          result = *(__int64 **)(a2 + 32);
          if ( *(__int64 **)(a2 + 24) == result )
          {
            result = (__int64 *)sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *(_BYTE *)result = 10;
            ++*(_QWORD *)(a2 + 32);
          }
          v16 += 2;
          if ( v16 == v15 )
            break;
          while ( 1 )
          {
            result = (__int64 *)*v16;
            if ( *v16 != -4096 && result != (__int64 *)-8192LL )
              break;
            v16 += 2;
            if ( v15 == v16 )
              return result;
          }
        }
        while ( v15 != v16 );
      }
    }
  }
  return result;
}
