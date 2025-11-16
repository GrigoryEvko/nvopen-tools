// Function: sub_266F010
// Address: 0x266f010
//
unsigned __int64 __fastcall sub_266F010(__int64 a1, __int64 *a2)
{
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  unsigned __int64 result; // rax
  int v8; // r8d
  const void *v9; // r9
  __int64 v10; // r8
  __int64 v11; // rsi
  int v12; // ecx
  __int64 *v13; // rdi
  __int64 v14; // r9
  __int64 v15; // rax
  _QWORD *v16; // rdi
  int v17; // edi
  int v18; // r10d

  if ( !*(_DWORD *)(a1 + 16) )
  {
    v4 = *(unsigned int *)(a1 + 40);
    v5 = *(_QWORD **)(a1 + 32);
    v6 = (__int64)&v5[v4];
    result = (unsigned __int64)sub_266E4D0(v5, v6, a2);
    if ( v6 == result )
      return result;
    v9 = (const void *)(result + 8);
    if ( v6 == result + 8 )
      goto LABEL_5;
LABEL_4:
    result = (unsigned __int64)memmove((void *)result, v9, v6 - (_QWORD)v9);
    v8 = *(_DWORD *)(a1 + 40);
LABEL_5:
    *(_DWORD *)(a1 + 40) = v8 - 1;
    return result;
  }
  result = *(unsigned int *)(a1 + 24);
  v10 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)result )
  {
    v11 = *a2;
    v12 = result - 1;
    result = ((_DWORD)result - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v13 = (__int64 *)(v10 + 8 * result);
    v14 = *v13;
    if ( v11 == *v13 )
    {
LABEL_9:
      *v13 = -8192;
      v15 = *(unsigned int *)(a1 + 40);
      --*(_DWORD *)(a1 + 16);
      v16 = *(_QWORD **)(a1 + 32);
      ++*(_DWORD *)(a1 + 20);
      v6 = (__int64)&v16[v15];
      result = (unsigned __int64)sub_266E4D0(v16, v6, a2);
      v9 = (const void *)(result + 8);
      if ( result + 8 == v6 )
        goto LABEL_5;
      goto LABEL_4;
    }
    v17 = 1;
    while ( v14 != -4096 )
    {
      v18 = v17 + 1;
      result = v12 & (unsigned int)(v17 + result);
      v13 = (__int64 *)(v10 + 8LL * (unsigned int)result);
      v14 = *v13;
      if ( v11 == *v13 )
        goto LABEL_9;
      v17 = v18;
    }
  }
  return result;
}
