// Function: sub_1A2EDE0
// Address: 0x1a2ede0
//
__int64 __fastcall sub_1A2EDE0(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  int v5; // r9d
  __int64 v6; // r8
  __int64 *v7; // r10
  int v8; // r11d
  __int64 result; // rax
  __int64 *v10; // rdi
  __int64 v11; // rcx
  int v12; // eax
  int v13; // edx
  __int64 *v14; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_19:
    v4 *= 2;
    goto LABEL_20;
  }
  v5 = v4 - 1;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v8 = 1;
  result = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v10 = (__int64 *)(v6 + 8 * result);
  v11 = *v10;
  if ( *a2 == *v10 )
    return result;
  while ( v11 != -8 )
  {
    if ( v11 != -16 || v7 )
      v10 = v7;
    result = v5 & (unsigned int)(v8 + result);
    v11 = *(_QWORD *)(v6 + 8LL * (unsigned int)result);
    if ( *a2 == v11 )
      return result;
    ++v8;
    v7 = v10;
    v10 = (__int64 *)(v6 + 8LL * (unsigned int)result);
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v10;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
    goto LABEL_19;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
LABEL_20:
    sub_1467110(a1, v4);
    sub_1463A20(a1, a2, &v14);
    v7 = v14;
    v13 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v7 = *a2;
  result = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 8, v6, v5);
    result = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = *a2;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
