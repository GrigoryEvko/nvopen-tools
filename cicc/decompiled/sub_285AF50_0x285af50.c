// Function: sub_285AF50
// Address: 0x285af50
//
__int64 __fastcall sub_285AF50(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  unsigned __int64 v7; // rcx
  __int64 v8; // r9
  __int64 v9; // r8
  unsigned int v10; // edx
  _QWORD *v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  __int64 result; // rax
  int v17; // eax
  int v18; // edx
  __int64 v19; // rax
  int v20; // eax
  int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rsi
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  unsigned int v28; // r15d
  _QWORD *v29; // rdi
  __int64 v30; // rcx

  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_27;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (_QWORD *)(v7 + 16LL * v10);
  v12 = *v11;
  if ( a2 == *v11 )
    goto LABEL_3;
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = (__int64)v11;
    v10 = (v6 - 1) & (v8 + v10);
    v11 = (_QWORD *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( a2 == *v11 )
      goto LABEL_3;
    v8 = (unsigned int)(v8 + 1);
  }
  v17 = *(_DWORD *)(a1 + 16);
  if ( v9 )
    v11 = (_QWORD *)v9;
  ++*(_QWORD *)a1;
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v6 )
  {
LABEL_27:
    sub_2854C20(a1, 2 * v6);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 8);
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *(_DWORD *)(a1 + 16) + 1;
      v11 = (_QWORD *)(v22 + 16LL * v23);
      v24 = *v11;
      if ( a2 != *v11 )
      {
        v8 = 1;
        v9 = 0;
        while ( v24 != -4096 )
        {
          if ( !v9 && v24 == -8192 )
            v9 = (__int64)v11;
          v23 = v21 & (v8 + v23);
          v11 = (_QWORD *)(v22 + 16LL * v23);
          v24 = *v11;
          if ( a2 == *v11 )
            goto LABEL_21;
          v8 = (unsigned int)(v8 + 1);
        }
        if ( v9 )
          v11 = (_QWORD *)v9;
      }
      goto LABEL_21;
    }
    goto LABEL_50;
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v18 <= v6 >> 3 )
  {
    sub_2854C20(a1, v6);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 8);
      v9 = 1;
      v28 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *(_DWORD *)(a1 + 16) + 1;
      v29 = 0;
      v11 = (_QWORD *)(v27 + 16LL * v28);
      v30 = *v11;
      if ( a2 != *v11 )
      {
        while ( v30 != -4096 )
        {
          if ( !v29 && v30 == -8192 )
            v29 = v11;
          v8 = (unsigned int)(v9 + 1);
          v28 = v26 & (v9 + v28);
          v11 = (_QWORD *)(v27 + 16LL * v28);
          v30 = *v11;
          if ( a2 == *v11 )
            goto LABEL_21;
          v9 = (unsigned int)v8;
        }
        if ( v29 )
          v11 = v29;
      }
      goto LABEL_21;
    }
LABEL_50:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_21:
  *(_DWORD *)(a1 + 16) = v18;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v11 = a2;
  v11[1] = 1;
  v19 = *(unsigned int *)(a1 + 40);
  v7 = *(unsigned int *)(a1 + 44);
  if ( v19 + 1 > v7 )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v19 + 1, 8u, v9, v8);
    v19 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v19) = a2;
  ++*(_DWORD *)(a1 + 40);
LABEL_3:
  v13 = v11[1];
  if ( (v13 & 1) != 0 )
    v14 = v13 >> 58;
  else
    v14 = *(unsigned int *)(v13 + 64);
  if ( a3 + 1 >= v14 )
    LODWORD(v14) = a3 + 1;
  sub_228BF90(v11 + 1, v14, 0, v7, v9, v8);
  v15 = v11[1];
  if ( (v15 & 1) != 0 )
  {
    result = 2 * ((v15 >> 58 << 57) | ~(-1LL << (v15 >> 58)) & (~(-1LL << (v15 >> 58)) & (v15 >> 1) | (1LL << a3))) + 1;
    v11[1] = result;
  }
  else
  {
    *(_QWORD *)(*(_QWORD *)v15 + 8LL * ((unsigned int)a3 >> 6)) |= 1LL << a3;
    return 1LL << a3;
  }
  return result;
}
