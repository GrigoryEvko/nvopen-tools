// Function: sub_D48300
// Address: 0xd48300
//
__int64 *__fastcall sub_D48300(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v3; // rcx
  unsigned int v6; // edx
  __int64 *v7; // r13
  __int64 v8; // rsi
  __int64 i; // r15
  __int64 v10; // rsi
  _QWORD *v11; // rdi
  char *v12; // rax
  char *v13; // rdx
  char *v14; // rsi
  bool v15; // zf
  __int64 *v16; // rdi
  __int64 *v17; // rdx
  __int64 v18; // rcx
  int v19; // r8d
  __int64 v20; // [rsp-40h] [rbp-40h] BYREF

  result = (__int64 *)*(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)result )
  {
    v6 = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v3 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      result = (__int64 *)(v3 + 16LL * (_QWORD)result);
      if ( v7 != result )
      {
        for ( i = v7[1]; i; i = *(_QWORD *)i )
        {
          v10 = *(_QWORD *)(i + 40);
          v11 = *(_QWORD **)(i + 32);
          v20 = a2;
          v12 = (char *)sub_D463D0(v11, v10, &v20);
          v13 = *(char **)(i + 40);
          v14 = v12 + 8;
          if ( v13 != v12 + 8 )
          {
            memmove(v12, v14, v13 - v14);
            v14 = *(char **)(i + 40);
          }
          v15 = *(_BYTE *)(i + 84) == 0;
          *(_QWORD *)(i + 40) = v14 - 8;
          if ( v15 )
          {
            result = sub_C8CA60(i + 56, v20);
            if ( result )
            {
              *result = -2;
              ++*(_DWORD *)(i + 80);
              ++*(_QWORD *)(i + 56);
            }
          }
          else
          {
            v16 = *(__int64 **)(i + 64);
            v17 = &v16[*(unsigned int *)(i + 76)];
            result = v16;
            if ( v16 != v17 )
            {
              while ( v20 != *result )
              {
                if ( v17 == ++result )
                  goto LABEL_13;
              }
              v18 = (unsigned int)(*(_DWORD *)(i + 76) - 1);
              *(_DWORD *)(i + 76) = v18;
              *result = v16[v18];
              ++*(_QWORD *)(i + 56);
            }
          }
LABEL_13:
          ;
        }
        *v7 = -8192;
        --*(_DWORD *)(a1 + 16);
        ++*(_DWORD *)(a1 + 20);
      }
    }
    else
    {
      v19 = 1;
      while ( v8 != -4096 )
      {
        v6 = ((_DWORD)result - 1) & (v19 + v6);
        v7 = (__int64 *)(v3 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        ++v19;
      }
    }
  }
  return result;
}
