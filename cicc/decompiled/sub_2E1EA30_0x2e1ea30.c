// Function: sub_2E1EA30
// Address: 0x2e1ea30
//
__int64 __fastcall sub_2E1EA30(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 result; // rax
  __int64 v6; // r9
  __int64 v7; // r8
  _DWORD *v8; // r10
  int v9; // r11d
  unsigned int v10; // edx
  _DWORD *v11; // rdi
  int v12; // ecx
  int v13; // eax
  int v14; // edx
  int v15; // r12d
  int v16; // eax
  int v17; // ecx
  unsigned int v18; // eax
  int v19; // edi
  int v20; // r11d
  int v21; // eax
  int v22; // ecx
  int v23; // r11d
  unsigned int v24; // eax
  int v25; // edi

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_19;
  }
  result = (unsigned int)*a2;
  v6 = v4 - 1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  v9 = 1;
  v10 = v6 & (37 * result);
  v11 = (_DWORD *)(v7 + 4LL * v10);
  v12 = *v11;
  if ( (_DWORD)result == *v11 )
    return result;
  while ( v12 != -1 )
  {
    if ( v8 || v12 != -2 )
      v11 = v8;
    v10 = v6 & (v9 + v10);
    v12 = *(_DWORD *)(v7 + 4LL * v10);
    if ( (_DWORD)result == v12 )
      return result;
    ++v9;
    v8 = v11;
    v11 = (_DWORD *)(v7 + 4LL * v10);
  }
  v13 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
    v8 = v11;
  ++*(_QWORD *)a1;
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_19:
    sub_A08C50(a1, 2 * v4);
    v16 = *(_DWORD *)(a1 + 24);
    if ( v16 )
    {
      v17 = v16 - 1;
      v7 = *(_QWORD *)(a1 + 8);
      v18 = (v16 - 1) & (37 * *a2);
      v8 = (_DWORD *)(v7 + 4LL * (v17 & (unsigned int)(37 * *a2)));
      v19 = *v8;
      v14 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v8 == *a2 )
        goto LABEL_13;
      v20 = 1;
      v6 = 0;
      while ( v19 != -1 )
      {
        if ( !v6 && v19 == -2 )
          v6 = (__int64)v8;
        v18 = v17 & (v20 + v18);
        v8 = (_DWORD *)(v7 + 4LL * v18);
        v19 = *v8;
        if ( *a2 == *v8 )
          goto LABEL_13;
        ++v20;
      }
LABEL_23:
      if ( v6 )
        v8 = (_DWORD *)v6;
      goto LABEL_13;
    }
LABEL_40:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v14 <= v4 >> 3 )
  {
    sub_A08C50(a1, v4);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v7 = *(_QWORD *)(a1 + 8);
      v6 = 0;
      v23 = 1;
      v24 = (v21 - 1) & (37 * *a2);
      v8 = (_DWORD *)(v7 + 4LL * (v22 & (unsigned int)(37 * *a2)));
      v25 = *v8;
      v14 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v8 == *a2 )
        goto LABEL_13;
      while ( v25 != -1 )
      {
        if ( v25 == -2 && !v6 )
          v6 = (__int64)v8;
        v24 = v22 & (v23 + v24);
        v8 = (_DWORD *)(v7 + 4LL * v24);
        v25 = *v8;
        if ( *a2 == *v8 )
          goto LABEL_13;
        ++v23;
      }
      goto LABEL_23;
    }
    goto LABEL_40;
  }
LABEL_13:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *v8 != -1 )
    --*(_DWORD *)(a1 + 20);
  v15 = *a2;
  *v8 = v15;
  result = *(unsigned int *)(a1 + 40);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 4u, v7, v6);
    result = *(unsigned int *)(a1 + 40);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * result) = v15;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
