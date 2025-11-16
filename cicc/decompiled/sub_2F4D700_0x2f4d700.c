// Function: sub_2F4D700
// Address: 0x2f4d700
//
unsigned __int64 __fastcall sub_2F4D700(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  unsigned __int64 result; // rax
  int v8; // r8d
  const void *v9; // r9
  __int64 v10; // rdi
  int v11; // edx
  __int64 *v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rax
  _QWORD *v15; // rdi
  int v16; // ecx
  int v17; // r9d
  __int64 v18[2]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_DWORD *)(a1 + 28968);
  v18[0] = a2;
  if ( !v3 )
  {
    v4 = *(unsigned int *)(a1 + 28992);
    v5 = *(_QWORD **)(a1 + 28984);
    v6 = (__int64)&v5[v4];
    result = (unsigned __int64)sub_2F4C750(v5, v6, v18);
    if ( v6 == result )
      return result;
    v9 = (const void *)(result + 8);
    if ( v6 == result + 8 )
      goto LABEL_5;
LABEL_4:
    result = (unsigned __int64)memmove((void *)result, v9, v6 - (_QWORD)v9);
    v8 = *(_DWORD *)(a1 + 28992);
LABEL_5:
    *(_DWORD *)(a1 + 28992) = v8 - 1;
    return result;
  }
  result = *(unsigned int *)(a1 + 28976);
  v10 = *(_QWORD *)(a1 + 28960);
  if ( (_DWORD)result )
  {
    v11 = result - 1;
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = (__int64 *)(v10 + 8 * result);
    v13 = *v12;
    if ( a2 == *v12 )
    {
LABEL_9:
      *v12 = -8192;
      v14 = *(unsigned int *)(a1 + 28992);
      --*(_DWORD *)(a1 + 28968);
      v15 = *(_QWORD **)(a1 + 28984);
      ++*(_DWORD *)(a1 + 28972);
      v6 = (__int64)&v15[v14];
      result = (unsigned __int64)sub_2F4C750(v15, v6, v18);
      v9 = (const void *)(result + 8);
      if ( result + 8 == v6 )
        goto LABEL_5;
      goto LABEL_4;
    }
    v16 = 1;
    while ( v13 != -4096 )
    {
      v17 = v16 + 1;
      result = v11 & (unsigned int)(v16 + result);
      v12 = (__int64 *)(v10 + 8LL * (unsigned int)result);
      v13 = *v12;
      if ( a2 == *v12 )
        goto LABEL_9;
      v16 = v17;
    }
  }
  return result;
}
