// Function: sub_2C89C70
// Address: 0x2c89c70
//
_QWORD *__fastcall sub_2C89C70(__int64 a1, int a2, __int64 a3)
{
  _QWORD *result; // rax
  int v5; // edx
  __int64 v7; // r12
  __int64 v8; // r9
  __int64 v9; // r14
  __int64 v10; // r10
  unsigned int v11; // edx
  _QWORD *v12; // rax
  __int64 v13; // r8
  unsigned int v14; // esi
  __int64 *v15; // r13
  int v16; // ecx
  int v17; // ecx
  __int64 v18; // r10
  unsigned int v19; // edx
  int v20; // eax
  _QWORD *v21; // rdi
  __int64 v22; // r8
  __int64 v23; // rax
  int v24; // eax
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // r10
  unsigned int v28; // edx
  __int64 v29; // r8
  _QWORD *v30; // r11
  __int64 v31; // [rsp+8h] [rbp-38h]
  int v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]
  int v34; // [rsp+8h] [rbp-38h]
  int v35; // [rsp+8h] [rbp-38h]

  result = *(_QWORD **)(a1 + 48);
  v5 = *((_DWORD *)result + 2);
  if ( v5 )
  {
    v7 = 0;
    v8 = a1;
    v9 = 8LL * (unsigned int)(v5 - 1);
    while ( 1 )
    {
      v14 = *(_DWORD *)(a3 + 24);
      v15 = (__int64 *)(v7 + *result);
      if ( !v14 )
        break;
      v10 = *(_QWORD *)(a3 + 8);
      v11 = (v14 - 1) & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
      v12 = (_QWORD *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( *v15 == *v12 )
      {
LABEL_4:
        result = v12 + 1;
        *(_DWORD *)result = a2;
        if ( v9 == v7 )
          return result;
        goto LABEL_5;
      }
      v32 = 1;
      v21 = 0;
      while ( v13 != -4096 )
      {
        if ( !v21 && v13 == -8192 )
          v21 = v12;
        v11 = (v14 - 1) & (v32 + v11);
        v12 = (_QWORD *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( *v15 == *v12 )
          goto LABEL_4;
        ++v32;
      }
      if ( !v21 )
        v21 = v12;
      v24 = *(_DWORD *)(a3 + 16);
      ++*(_QWORD *)a3;
      v20 = v24 + 1;
      if ( 4 * v20 >= 3 * v14 )
        goto LABEL_8;
      if ( v14 - *(_DWORD *)(a3 + 20) - v20 <= v14 >> 3 )
      {
        v33 = v8;
        sub_9BAAD0(a3, v14);
        v25 = *(_DWORD *)(a3 + 24);
        if ( !v25 )
        {
LABEL_44:
          ++*(_DWORD *)(a3 + 16);
          BUG();
        }
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a3 + 8);
        v8 = v33;
        v28 = v26 & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
        v20 = *(_DWORD *)(a3 + 16) + 1;
        v21 = (_QWORD *)(v27 + 16LL * v28);
        v29 = *v21;
        if ( *v21 != *v15 )
        {
          v34 = 1;
          v30 = 0;
          while ( v29 != -4096 )
          {
            if ( !v30 && v29 == -8192 )
              v30 = v21;
            v28 = v26 & (v34 + v28);
            v21 = (_QWORD *)(v27 + 16LL * v28);
            v29 = *v21;
            if ( *v15 == *v21 )
              goto LABEL_10;
            ++v34;
          }
          goto LABEL_24;
        }
      }
LABEL_10:
      *(_DWORD *)(a3 + 16) = v20;
      if ( *v21 != -4096 )
        --*(_DWORD *)(a3 + 20);
      v23 = *v15;
      *((_DWORD *)v21 + 2) = 0;
      *v21 = v23;
      result = v21 + 1;
      *((_DWORD *)v21 + 2) = a2;
      if ( v9 == v7 )
        return result;
LABEL_5:
      result = *(_QWORD **)(v8 + 48);
      v7 += 8;
    }
    ++*(_QWORD *)a3;
LABEL_8:
    v31 = v8;
    sub_9BAAD0(a3, 2 * v14);
    v16 = *(_DWORD *)(a3 + 24);
    if ( !v16 )
      goto LABEL_44;
    v17 = v16 - 1;
    v18 = *(_QWORD *)(a3 + 8);
    v8 = v31;
    v19 = v17 & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
    v20 = *(_DWORD *)(a3 + 16) + 1;
    v21 = (_QWORD *)(v18 + 16LL * v19);
    v22 = *v21;
    if ( *v21 != *v15 )
    {
      v35 = 1;
      v30 = 0;
      while ( v22 != -4096 )
      {
        if ( !v30 && v22 == -8192 )
          v30 = v21;
        v19 = v17 & (v35 + v19);
        v21 = (_QWORD *)(v18 + 16LL * v19);
        v22 = *v21;
        if ( *v15 == *v21 )
          goto LABEL_10;
        ++v35;
      }
LABEL_24:
      if ( v30 )
        v21 = v30;
      goto LABEL_10;
    }
    goto LABEL_10;
  }
  return result;
}
