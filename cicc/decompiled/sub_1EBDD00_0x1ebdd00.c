// Function: sub_1EBDD00
// Address: 0x1ebdd00
//
unsigned __int64 __fastcall sub_1EBDD00(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // ecx
  unsigned __int64 result; // rax
  __int64 *v6; // rdx
  __int64 v7; // r8
  unsigned int v8; // eax
  _QWORD *v9; // rdi
  __int64 v10; // rsi
  int v11; // r8d
  int v12; // edx
  int v13; // r9d
  __int64 v14; // [rsp+8h] [rbp-18h] BYREF

  v14 = a2;
  if ( (*(_BYTE *)(a1 + 27424) & 1) != 0 )
  {
    v3 = a1 + 27432;
    v4 = 7;
  }
  else
  {
    result = *(unsigned int *)(a1 + 27440);
    v3 = *(_QWORD *)(a1 + 27432);
    if ( !(_DWORD)result )
      return result;
    v4 = result - 1;
  }
  result = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 8 * result);
  v7 = *v6;
  if ( a2 == *v6 )
  {
LABEL_4:
    *v6 = -16;
    v8 = *(_DWORD *)(a1 + 27424);
    ++*(_DWORD *)(a1 + 27428);
    v9 = *(_QWORD **)(a1 + 27496);
    *(_DWORD *)(a1 + 27424) = (2 * (v8 >> 1) - 2) | v8 & 1;
    v10 = (__int64)&v9[*(unsigned int *)(a1 + 27504)];
    result = (unsigned __int64)sub_1EBB320(v9, v10, &v14);
    if ( result + 8 != v10 )
    {
      result = (unsigned __int64)memmove((void *)result, (const void *)(result + 8), v10 - (result + 8));
      v11 = *(_DWORD *)(a1 + 27504);
    }
    *(_DWORD *)(a1 + 27504) = v11 - 1;
  }
  else
  {
    v12 = 1;
    while ( v7 != -8 )
    {
      v13 = v12 + 1;
      result = v4 & (unsigned int)(v12 + result);
      v6 = (__int64 *)(v3 + 8LL * (unsigned int)result);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_4;
      v12 = v13;
    }
  }
  return result;
}
