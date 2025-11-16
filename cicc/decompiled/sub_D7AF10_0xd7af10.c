// Function: sub_D7AF10
// Address: 0xd7af10
//
unsigned __int64 __fastcall sub_D7AF10(__int64 a1, _QWORD *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 v6; // r8
  _QWORD *v7; // r10
  int v8; // r11d
  unsigned __int64 result; // rax
  unsigned int v10; // edi
  _QWORD *v11; // rcx
  unsigned __int64 v12; // rdx
  int v13; // eax
  int v14; // edx
  __int64 v15; // r12
  int v16; // eax
  int v17; // ecx
  unsigned __int64 v18; // rsi
  unsigned int v19; // edi
  unsigned __int64 v20; // rax
  int v21; // r11d
  int v22; // eax
  int v23; // ecx
  int v24; // r11d
  unsigned __int64 v25; // rsi
  unsigned int v26; // edi
  unsigned __int64 v27; // rax

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
  result = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = result & (v4 - 1);
  v11 = (_QWORD *)(v6 + 8LL * v10);
  v12 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
  if ( result == v12 )
    return result;
  while ( v12 != -8 )
  {
    if ( v7 || v12 != -16 )
      v11 = v7;
    v10 = v5 & (v8 + v10);
    v12 = *(_QWORD *)(v6 + 8LL * v10) & 0xFFFFFFFFFFFFFFF8LL;
    if ( result == v12 )
      return result;
    ++v8;
    v7 = v11;
    v11 = (_QWORD *)(v6 + 8LL * v10);
  }
  v13 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v11;
  ++*(_QWORD *)a1;
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_19:
    sub_BB0720(a1, 2 * v4);
    v16 = *(_DWORD *)(a1 + 24);
    if ( v16 )
    {
      v17 = v16 - 1;
      v6 = *(_QWORD *)(a1 + 8);
      v18 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
      v19 = v18 & (v16 - 1);
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v7 = (_QWORD *)(v6 + 8LL * v19);
      v20 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v18 == v20 )
        goto LABEL_13;
      v21 = 1;
      v5 = 0;
      while ( v20 != -8 )
      {
        if ( !v5 && v20 == -16 )
          v5 = (__int64)v7;
        v19 = v17 & (v21 + v19);
        v7 = (_QWORD *)(v6 + 8LL * v19);
        v20 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v18 == v20 )
          goto LABEL_13;
        ++v21;
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
  if ( v4 - *(_DWORD *)(a1 + 20) - v14 <= v4 >> 3 )
  {
    sub_BB0720(a1, v4);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v6 = *(_QWORD *)(a1 + 8);
      v5 = 0;
      v24 = 1;
      v25 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
      v26 = v25 & (v22 - 1);
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v7 = (_QWORD *)(v6 + 8LL * v26);
      v27 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v25 == v27 )
        goto LABEL_13;
      while ( v27 != -8 )
      {
        if ( v27 == -16 && !v5 )
          v5 = (__int64)v7;
        v26 = v23 & (v24 + v26);
        v7 = (_QWORD *)(v6 + 8LL * v26);
        v27 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v25 == v27 )
          goto LABEL_13;
        ++v24;
      }
      goto LABEL_23;
    }
    goto LABEL_40;
  }
LABEL_13:
  *(_DWORD *)(a1 + 16) = v14;
  if ( (*v7 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
    --*(_DWORD *)(a1 + 20);
  *v7 = *a2;
  result = *(unsigned int *)(a1 + 40);
  v15 = *a2;
  if ( result + 1 > *(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, v6, v5);
    result = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v15;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
