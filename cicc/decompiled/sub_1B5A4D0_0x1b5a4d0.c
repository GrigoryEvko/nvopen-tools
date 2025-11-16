// Function: sub_1B5A4D0
// Address: 0x1b5a4d0
//
__int64 __fastcall sub_1B5A4D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  char v8; // dl
  __int64 v9; // rdi
  int v10; // esi
  __int64 result; // rax
  unsigned int v12; // esi
  unsigned int v13; // eax
  __int64 *v14; // r10
  int v15; // ecx
  unsigned int v16; // edi
  int v17; // r11d
  __int64 *v18; // [rsp+8h] [rbp-28h] BYREF

  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( v8 )
  {
    v9 = a1 + 16;
    v10 = 3;
  }
  else
  {
    v12 = *(_DWORD *)(a1 + 24);
    v9 = *(_QWORD *)(a1 + 16);
    if ( !v12 )
    {
      v13 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v14 = 0;
      v15 = (v13 >> 1) + 1;
LABEL_8:
      v16 = 3 * v12;
      goto LABEL_9;
    }
    v10 = v12 - 1;
  }
  result = v10 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  a6 = (__int64 *)(v9 + 8 * result);
  a5 = *a6;
  if ( *a2 == *a6 )
    return result;
  v17 = 1;
  v14 = 0;
  while ( a5 != -8 )
  {
    if ( v14 || a5 != -16 )
      a6 = v14;
    result = v10 & (unsigned int)(v17 + result);
    a5 = *(_QWORD *)(v9 + 8LL * (unsigned int)result);
    if ( *a2 == a5 )
      return result;
    ++v17;
    v14 = a6;
    a6 = (__int64 *)(v9 + 8LL * (unsigned int)result);
  }
  v13 = *(_DWORD *)(a1 + 8);
  if ( !v14 )
    v14 = a6;
  ++*(_QWORD *)a1;
  v15 = (v13 >> 1) + 1;
  if ( !v8 )
  {
    v12 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v16 = 12;
  v12 = 4;
LABEL_9:
  if ( 4 * v15 >= v16 )
  {
    v12 *= 2;
    goto LABEL_23;
  }
  if ( v12 - *(_DWORD *)(a1 + 12) - v15 <= v12 >> 3 )
  {
LABEL_23:
    sub_1918F70(a1, v12);
    sub_1A54750(a1, a2, &v18);
    v14 = v18;
    v13 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = (2 * (v13 >> 1) + 2) | v13 & 1;
  if ( *v14 != -8 )
    --*(_DWORD *)(a1 + 12);
  *v14 = *a2;
  result = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 60) )
  {
    sub_16CD150(a1 + 48, (const void *)(a1 + 64), 0, 8, a5, (int)a6);
    result = *(unsigned int *)(a1 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * result) = *a2;
  ++*(_DWORD *)(a1 + 56);
  return result;
}
