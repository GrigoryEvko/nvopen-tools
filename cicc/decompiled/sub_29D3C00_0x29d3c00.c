// Function: sub_29D3C00
// Address: 0x29d3c00
//
__int64 __fastcall sub_29D3C00(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 v6; // r8
  _QWORD *v7; // r10
  int v8; // r11d
  __int64 result; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  int v12; // eax
  int v13; // edx
  __int64 v14; // r12
  int v15; // eax
  int v16; // ecx
  unsigned int v17; // eax
  __int64 v18; // rdi
  int v19; // r11d
  int v20; // eax
  int v21; // ecx
  int v22; // r11d
  unsigned int v23; // eax
  __int64 v24; // rdi

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_19;
  }
  v5 = v4 - 1;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v8 = 1;
  result = (unsigned int)v5 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v10 = (_QWORD *)(v6 + 8 * result);
  v11 = *v10;
  if ( *a2 == *v10 )
    return result;
  while ( v11 != -4096 )
  {
    if ( v7 || v11 != -8192 )
      v10 = v7;
    result = (unsigned int)v5 & (v8 + (_DWORD)result);
    v11 = *(_QWORD *)(v6 + 8LL * (unsigned int)result);
    if ( *a2 == v11 )
      return result;
    ++v8;
    v7 = v10;
    v10 = (_QWORD *)(v6 + 8LL * (unsigned int)result);
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v10;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_19:
    sub_CF28B0(a1, 2 * v4);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v6 = *(_QWORD *)(a1 + 8);
      v17 = (v15 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (_QWORD *)(v6 + 8LL * v17);
      v18 = *v7;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v7 == *a2 )
        goto LABEL_13;
      v19 = 1;
      v5 = 0;
      while ( v18 != -4096 )
      {
        if ( !v5 && v18 == -8192 )
          v5 = (__int64)v7;
        v17 = v16 & (v19 + v17);
        v7 = (_QWORD *)(v6 + 8LL * v17);
        v18 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_13;
        ++v19;
      }
LABEL_23:
      if ( v5 )
        v7 = (_QWORD *)v5;
      goto LABEL_13;
    }
LABEL_40:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_CF28B0(a1, v4);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v6 = *(_QWORD *)(a1 + 8);
      v5 = 0;
      v22 = 1;
      v23 = (v20 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (_QWORD *)(v6 + 8LL * v23);
      v24 = *v7;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v7 == *a2 )
        goto LABEL_13;
      while ( v24 != -4096 )
      {
        if ( v24 == -8192 && !v5 )
          v5 = (__int64)v7;
        v23 = v21 & (v22 + v23);
        v7 = (_QWORD *)(v6 + 8LL * v23);
        v24 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_13;
        ++v22;
      }
      goto LABEL_23;
    }
    goto LABEL_40;
  }
LABEL_13:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *v7 = v14;
  result = *(unsigned int *)(a1 + 40);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, v6, v5);
    result = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v14;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
