// Function: sub_2921310
// Address: 0x2921310
//
__int64 *__fastcall sub_2921310(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v4; // rdx
  __int64 *v5; // r13
  __int64 v6; // rsi
  __int64 *v7; // rbx
  int v8; // ecx
  _QWORD *v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // rax
  int v12; // r8d
  _QWORD *v13; // rdi
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rcx
  _QWORD *v17; // rdi
  __int64 *v18; // rax
  __int64 v19[5]; // [rsp+8h] [rbp-28h] BYREF

  result = *(__int64 **)(a2 + 8);
  if ( *(_BYTE *)(a2 + 28) )
    v4 = *(unsigned int *)(a2 + 20);
  else
    v4 = *(unsigned int *)(a2 + 16);
  v5 = &result[v4];
  if ( result != v5 )
  {
    while ( 1 )
    {
      v6 = *result;
      v7 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v5 == ++result )
        return result;
    }
    if ( v5 != result )
    {
      v19[0] = *result;
      v8 = *(_DWORD *)(a1 + 20);
      if ( v8 != *(_DWORD *)(a1 + 24) )
        goto LABEL_19;
LABEL_9:
      v9 = *(_QWORD **)(a1 + 160);
      v10 = (__int64)&v9[*(unsigned int *)(a1 + 168)];
      v11 = sub_2912630(v9, v10, v19);
      if ( (_QWORD *)v10 == v11 )
        goto LABEL_13;
      if ( (_QWORD *)v10 != v11 + 1 )
      {
LABEL_11:
        memmove(v11, v11 + 1, v10 - (_QWORD)(v11 + 1));
        v12 = *(_DWORD *)(a1 + 168);
      }
LABEL_12:
      *(_DWORD *)(a1 + 168) = v12 - 1;
      while ( 1 )
      {
        while ( 1 )
        {
LABEL_13:
          result = v7 + 1;
          if ( v7 + 1 == v5 )
            return result;
          v6 = *result;
          for ( ++v7; (unsigned __int64)*result >= 0xFFFFFFFFFFFFFFFELL; v7 = result )
          {
            if ( v5 == ++result )
              return result;
            v6 = *result;
          }
          if ( v5 == v7 )
            return result;
          v19[0] = v6;
          v8 = *(_DWORD *)(a1 + 20);
          if ( v8 == *(_DWORD *)(a1 + 24) )
            goto LABEL_9;
LABEL_19:
          if ( !*(_BYTE *)(a1 + 28) )
            break;
          v13 = *(_QWORD **)(a1 + 8);
          v14 = &v13[v8];
          if ( v13 != v14 )
          {
            v15 = *(_QWORD **)(a1 + 8);
            while ( v6 != *v15 )
            {
              if ( v14 == ++v15 )
                goto LABEL_13;
            }
            v16 = (unsigned int)(v8 - 1);
            *(_DWORD *)(a1 + 20) = v16;
            *v15 = v13[v16];
            ++*(_QWORD *)a1;
LABEL_25:
            v17 = *(_QWORD **)(a1 + 160);
            v10 = (__int64)&v17[*(unsigned int *)(a1 + 168)];
            v11 = sub_2912630(v17, v10, v19);
            if ( v11 + 1 == (_QWORD *)v10 )
              goto LABEL_12;
            goto LABEL_11;
          }
        }
        v18 = sub_C8CA60(a1, v6);
        if ( v18 )
        {
          *v18 = -2;
          ++*(_DWORD *)(a1 + 24);
          ++*(_QWORD *)a1;
          goto LABEL_25;
        }
      }
    }
  }
  return result;
}
