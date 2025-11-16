// Function: sub_27915B0
// Address: 0x27915b0
//
_QWORD *__fastcall sub_27915B0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  unsigned int v8; // esi
  __int64 v9; // r8
  int v10; // r10d
  unsigned int v11; // edi
  int *v12; // r15
  _DWORD *v13; // rax
  int v14; // edx
  _QWORD *result; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdi
  unsigned int v19; // edx
  __int64 v20; // rcx
  int v21; // edi
  int v22; // edi
  __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned int v25; // edx
  __int64 v26; // rcx
  int v27; // eax
  int v28; // ecx
  __int64 v29; // r8
  unsigned int v30; // edx
  int v31; // esi
  int v32; // r10d
  _DWORD *v33; // r9
  int v34; // eax
  int v35; // edx
  __int64 v36; // r8
  int v37; // r10d
  unsigned int v38; // ecx
  int v39; // esi

  v8 = *(_DWORD *)(a1 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_34;
  }
  v9 = *(_QWORD *)(a1 + 8);
  v10 = 1;
  v11 = (v8 - 1) & (37 * a2);
  v12 = (int *)(v9 + 40LL * v11);
  v13 = 0;
  v14 = *v12;
  if ( *v12 != a2 )
  {
    while ( v14 != -1 )
    {
      if ( !v13 && v14 == -2 )
        v13 = v12;
      v11 = (v8 - 1) & (v10 + v11);
      v12 = (int *)(v9 + 40LL * v11);
      v14 = *v12;
      if ( *v12 == a2 )
        goto LABEL_3;
      ++v10;
    }
    v21 = *(_DWORD *)(a1 + 16);
    if ( !v13 )
      v13 = v12;
    ++*(_QWORD *)a1;
    v22 = v21 + 1;
    if ( 4 * v22 < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a1 + 20) - v22 > v8 >> 3 )
      {
LABEL_23:
        *(_DWORD *)(a1 + 16) = v22;
        if ( *v13 != -1 )
          --*(_DWORD *)(a1 + 20);
        *v13 = a2;
        result = v13 + 2;
        *result = 0;
        result[1] = 0;
        result[2] = 0;
        result[3] = 0;
        goto LABEL_26;
      }
      sub_27913C0(a1, v8);
      v34 = *(_DWORD *)(a1 + 24);
      if ( v34 )
      {
        v35 = v34 - 1;
        v36 = *(_QWORD *)(a1 + 8);
        v37 = 1;
        v33 = 0;
        v38 = (v34 - 1) & (37 * a2);
        v22 = *(_DWORD *)(a1 + 16) + 1;
        v13 = (_DWORD *)(v36 + 40LL * v38);
        v39 = *v13;
        if ( *v13 == a2 )
          goto LABEL_23;
        while ( v39 != -1 )
        {
          if ( !v33 && v39 == -2 )
            v33 = v13;
          v38 = v35 & (v37 + v38);
          v13 = (_DWORD *)(v36 + 40LL * v38);
          v39 = *v13;
          if ( *v13 == a2 )
            goto LABEL_23;
          ++v37;
        }
        goto LABEL_38;
      }
      goto LABEL_54;
    }
LABEL_34:
    sub_27913C0(a1, 2 * v8);
    v27 = *(_DWORD *)(a1 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 8);
      v30 = (v27 - 1) & (37 * a2);
      v22 = *(_DWORD *)(a1 + 16) + 1;
      v13 = (_DWORD *)(v29 + 40LL * v30);
      v31 = *v13;
      if ( *v13 == a2 )
        goto LABEL_23;
      v32 = 1;
      v33 = 0;
      while ( v31 != -1 )
      {
        if ( !v33 && v31 == -2 )
          v33 = v13;
        v30 = v28 & (v32 + v30);
        v13 = (_DWORD *)(v29 + 40LL * v30);
        v31 = *v13;
        if ( *v13 == a2 )
          goto LABEL_23;
        ++v32;
      }
LABEL_38:
      if ( v33 )
        v13 = v33;
      goto LABEL_23;
    }
LABEL_54:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_3:
  result = v12 + 2;
  if ( *((_QWORD *)v12 + 1) )
  {
    v16 = *(_QWORD *)(a1 + 32);
    *(_QWORD *)(a1 + 112) += 32LL;
    result = (_QWORD *)((v16 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( *(_QWORD *)(a1 + 40) >= (unsigned __int64)(result + 4) && v16 )
      *(_QWORD *)(a1 + 32) = result + 4;
    else
      result = (_QWORD *)sub_9D1E70(a1 + 32, 32, 32, 3);
    *result = a3;
    result[1] = a4;
    result[3] = *((_QWORD *)v12 + 4);
    v17 = *(_QWORD *)(a1 + 128);
    if ( a4 )
    {
      v18 = (unsigned int)(*(_DWORD *)(a4 + 44) + 1);
      v19 = *(_DWORD *)(a4 + 44) + 1;
    }
    else
    {
      v18 = 0;
      v19 = 0;
    }
    v20 = 0;
    if ( v19 < *(_DWORD *)(v17 + 32) )
      v20 = *(_QWORD *)(*(_QWORD *)(v17 + 24) + 8 * v18);
    result[2] = v20;
    *((_QWORD *)v12 + 4) = result;
    return result;
  }
LABEL_26:
  *result = a3;
  result[1] = a4;
  v23 = *(_QWORD *)(a1 + 128);
  if ( a4 )
  {
    v24 = (unsigned int)(*(_DWORD *)(a4 + 44) + 1);
    v25 = *(_DWORD *)(a4 + 44) + 1;
  }
  else
  {
    v24 = 0;
    v25 = 0;
  }
  v26 = 0;
  if ( v25 < *(_DWORD *)(v23 + 32) )
    v26 = *(_QWORD *)(*(_QWORD *)(v23 + 24) + 8 * v24);
  result[2] = v26;
  return result;
}
