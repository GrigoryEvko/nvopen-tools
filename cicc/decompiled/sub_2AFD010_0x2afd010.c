// Function: sub_2AFD010
// Address: 0x2afd010
//
__int64 *__fastcall sub_2AFD010(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  __int64 *result; // rax
  int v11; // edx
  __int64 v12; // r12
  unsigned int v13; // esi
  int v14; // r15d
  __int64 v15; // r8
  __int64 *v16; // r11
  unsigned int v17; // edi
  _QWORD *v18; // rdx
  __int64 v19; // rcx
  __int64 **v20; // rdx
  int v21; // r10d
  int v22; // edi
  int v23; // ecx
  int v24; // edx
  int v25; // edx
  __int64 v26; // r8
  unsigned int v27; // esi
  __int64 v28; // rdi
  int v29; // r10d
  __int64 *v30; // r9
  int v31; // edx
  int v32; // esi
  __int64 v33; // rdi
  __int64 *v34; // r8
  unsigned int v35; // r13d
  int v36; // r9d
  __int64 v37; // rdx
  __int64 *v38; // [rsp+8h] [rbp-38h]
  __int64 *v39; // [rsp+8h] [rbp-38h]

  v4 = *a1;
  v5 = *(unsigned int *)(*a1 + 264);
  v6 = *(_QWORD *)(*a1 + 248);
  if ( (_DWORD)v5 )
  {
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
        return (__int64 *)v8[1];
    }
    else
    {
      v11 = 1;
      while ( v9 != -4096 )
      {
        v21 = v11 + 1;
        v7 = (v5 - 1) & (v11 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
          goto LABEL_3;
        v11 = v21;
      }
    }
  }
  result = sub_DD8400(*(_QWORD *)(v4 + 32), a2);
  v12 = *a1;
  v13 = *(_DWORD *)(v12 + 264);
  if ( !v13 )
  {
    ++*(_QWORD *)(v12 + 240);
    goto LABEL_27;
  }
  v14 = 1;
  v15 = *(_QWORD *)(v12 + 248);
  v16 = 0;
  v17 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v18 = (_QWORD *)(v15 + 16LL * v17);
  v19 = *v18;
  if ( *v18 != a2 )
  {
    while ( v19 != -4096 )
    {
      if ( v19 == -8192 && !v16 )
        v16 = v18;
      v17 = (v13 - 1) & (v14 + v17);
      v18 = (_QWORD *)(v15 + 16LL * v17);
      v19 = *v18;
      if ( *v18 == a2 )
        goto LABEL_9;
      ++v14;
    }
    v22 = *(_DWORD *)(v12 + 256);
    if ( !v16 )
      v16 = v18;
    ++*(_QWORD *)(v12 + 240);
    v23 = v22 + 1;
    if ( 4 * (v22 + 1) < 3 * v13 )
    {
      if ( v13 - *(_DWORD *)(v12 + 260) - v23 > v13 >> 3 )
      {
LABEL_23:
        *(_DWORD *)(v12 + 256) = v23;
        if ( *v16 != -4096 )
          --*(_DWORD *)(v12 + 260);
        *v16 = a2;
        v20 = (__int64 **)(v16 + 1);
        v16[1] = 0;
        goto LABEL_10;
      }
      v39 = result;
      sub_D3AEE0(v12 + 240, v13);
      v31 = *(_DWORD *)(v12 + 264);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(v12 + 248);
        v34 = 0;
        v35 = (v31 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v36 = 1;
        v23 = *(_DWORD *)(v12 + 256) + 1;
        result = v39;
        v16 = (__int64 *)(v33 + 16LL * v35);
        v37 = *v16;
        if ( *v16 != a2 )
        {
          while ( v37 != -4096 )
          {
            if ( v37 == -8192 && !v34 )
              v34 = v16;
            v35 = v32 & (v36 + v35);
            v16 = (__int64 *)(v33 + 16LL * v35);
            v37 = *v16;
            if ( *v16 == a2 )
              goto LABEL_23;
            ++v36;
          }
          if ( v34 )
            v16 = v34;
        }
        goto LABEL_23;
      }
LABEL_50:
      ++*(_DWORD *)(v12 + 256);
      BUG();
    }
LABEL_27:
    v38 = result;
    sub_D3AEE0(v12 + 240, 2 * v13);
    v24 = *(_DWORD *)(v12 + 264);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(v12 + 248);
      v27 = v25 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(v12 + 256) + 1;
      result = v38;
      v16 = (__int64 *)(v26 + 16LL * v27);
      v28 = *v16;
      if ( *v16 != a2 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !v30 )
            v30 = v16;
          v27 = v25 & (v29 + v27);
          v16 = (__int64 *)(v26 + 16LL * v27);
          v28 = *v16;
          if ( *v16 == a2 )
            goto LABEL_23;
          ++v29;
        }
        if ( v30 )
          v16 = v30;
      }
      goto LABEL_23;
    }
    goto LABEL_50;
  }
LABEL_9:
  v20 = (__int64 **)(v18 + 1);
LABEL_10:
  *v20 = result;
  return result;
}
