// Function: sub_B70050
// Address: 0xb70050
//
unsigned __int64 __fastcall sub_B70050(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // r8
  int v8; // r11d
  __int64 v9; // r9
  _QWORD *v10; // rdx
  unsigned int v11; // edi
  unsigned __int64 result; // rax
  __int64 v13; // rcx
  _BYTE *v14; // rdx
  _BYTE *v15; // rsi
  _QWORD *v16; // r12
  __int64 v17; // rdi
  int v18; // eax
  int v19; // ecx
  _BYTE *v20; // rdi
  size_t v21; // rdx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r8
  __int64 v25; // rdi
  int v26; // r10d
  _QWORD *v27; // r9
  int v28; // eax
  int v29; // r9d
  _QWORD *v30; // r8
  __int64 v31; // rdi
  unsigned int v32; // r14d
  __int64 v33; // rsi

  v5 = *a1;
  v6 = *(_DWORD *)(*a1 + 3488);
  v7 = *a1 + 3464;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 3464);
    goto LABEL_31;
  }
  v8 = 1;
  v9 = *(_QWORD *)(v5 + 3472);
  v10 = 0;
  v11 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v9 + 40LL * v11;
  v13 = *(_QWORD *)result;
  if ( a2 != *(_QWORD *)result )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v10 )
        v10 = (_QWORD *)result;
      v11 = (v6 - 1) & (v8 + v11);
      result = v9 + 40LL * v11;
      v13 = *(_QWORD *)result;
      if ( a2 == *(_QWORD *)result )
        goto LABEL_3;
      ++v8;
    }
    if ( !v10 )
      v10 = (_QWORD *)result;
    v18 = *(_DWORD *)(v5 + 3480);
    ++*(_QWORD *)(v5 + 3464);
    v19 = v18 + 1;
    if ( 4 * (v18 + 1) < 3 * v6 )
    {
      result = v6 - *(_DWORD *)(v5 + 3484) - v19;
      if ( (unsigned int)result > v6 >> 3 )
        goto LABEL_18;
      sub_B6FDE0(v7, v6);
      v28 = *(_DWORD *)(v5 + 3488);
      if ( v28 )
      {
        result = (unsigned int)(v28 - 1);
        v29 = 1;
        v30 = 0;
        v31 = *(_QWORD *)(v5 + 3472);
        v32 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v19 = *(_DWORD *)(v5 + 3480) + 1;
        v10 = (_QWORD *)(v31 + 40LL * v32);
        v33 = *v10;
        if ( a2 != *v10 )
        {
          while ( v33 != -4096 )
          {
            if ( !v30 && v33 == -8192 )
              v30 = v10;
            v32 = result & (v29 + v32);
            v10 = (_QWORD *)(v31 + 40LL * v32);
            v33 = *v10;
            if ( a2 == *v10 )
              goto LABEL_18;
            ++v29;
          }
          if ( v30 )
            v10 = v30;
        }
        goto LABEL_18;
      }
      goto LABEL_56;
    }
LABEL_31:
    sub_B6FDE0(v7, 2 * v6);
    v22 = *(_DWORD *)(v5 + 3488);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v5 + 3472);
      result = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(v5 + 3480) + 1;
      v10 = (_QWORD *)(v24 + 40 * result);
      v25 = *v10;
      if ( a2 != *v10 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -4096 )
        {
          if ( !v27 && v25 == -8192 )
            v27 = v10;
          result = v23 & (unsigned int)(v26 + result);
          v10 = (_QWORD *)(v24 + 40LL * (unsigned int)result);
          v25 = *v10;
          if ( a2 == *v10 )
            goto LABEL_18;
          ++v26;
        }
        if ( v27 )
          v10 = v27;
      }
LABEL_18:
      *(_DWORD *)(v5 + 3480) = v19;
      if ( *v10 != -4096 )
        --*(_DWORD *)(v5 + 3484);
      v20 = v10 + 3;
      *v10 = a2;
      v16 = v10 + 1;
      v10[1] = v10 + 3;
      v15 = a3 + 2;
      v10[2] = 0;
      *((_BYTE *)v10 + 24) = 0;
      v14 = (_BYTE *)*a3;
      if ( (_QWORD *)*a3 != a3 + 2 )
        goto LABEL_27;
      goto LABEL_21;
    }
LABEL_56:
    ++*(_DWORD *)(v5 + 3480);
    BUG();
  }
LABEL_3:
  v14 = (_BYTE *)*a3;
  v15 = *(_BYTE **)(result + 8);
  v16 = (_QWORD *)(result + 8);
  if ( (_QWORD *)*a3 != a3 + 2 )
  {
    if ( (_BYTE *)(result + 24) != v15 )
    {
      *(_QWORD *)(result + 8) = v14;
      v17 = *(_QWORD *)(result + 24);
      *(_QWORD *)(result + 16) = a3[1];
      *(_QWORD *)(result + 24) = a3[2];
      if ( v15 )
      {
        *a3 = v15;
        a3[2] = v17;
        goto LABEL_7;
      }
      v15 = a3 + 2;
LABEL_28:
      *a3 = v15;
      goto LABEL_7;
    }
    v15 = a3 + 2;
LABEL_27:
    *v16 = v14;
    v16[1] = a3[1];
    result = a3[2];
    v16[2] = result;
    goto LABEL_28;
  }
  v20 = *(_BYTE **)(result + 8);
  v15 = (_BYTE *)*a3;
LABEL_21:
  v21 = a3[1];
  if ( v21 )
  {
    if ( v21 == 1 )
    {
      result = *((unsigned __int8 *)a3 + 16);
      *v20 = result;
    }
    else
    {
      result = (unsigned __int64)memcpy(v20, v15, v21);
    }
    v21 = a3[1];
    v20 = (_BYTE *)*v16;
  }
  v16[1] = v21;
  v20[v21] = 0;
  v15 = (_BYTE *)*a3;
LABEL_7:
  a3[1] = 0;
  *v15 = 0;
  return result;
}
