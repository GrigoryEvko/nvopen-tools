// Function: sub_2805C60
// Address: 0x2805c60
//
__int64 *__fastcall sub_2805C60(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 *v10; // rax
  __int64 v11; // r13
  unsigned int v12; // r8d
  int v13; // r14d
  int v14; // r14d
  __int64 v15; // r15
  int v16; // eax
  int v17; // edx
  __int64 *v18; // rcx
  unsigned int v19; // eax
  __int64 *v20; // rsi
  __int64 v21; // rdi
  unsigned int v22; // eax
  unsigned int v23; // r14d
  int v24; // eax
  int v25; // ecx
  unsigned int i; // edx
  __int64 *result; // rax
  __int64 v28; // rdi
  unsigned int v29; // edx
  int v30; // eax
  int v31; // edx
  __int64 v32; // rdx
  int v33; // eax
  int v34; // r8d
  unsigned int j; // ecx
  __int64 v36; // r9
  unsigned int v37; // ecx
  unsigned int v38; // [rsp+0h] [rbp-60h]
  __int64 v39; // [rsp+0h] [rbp-60h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  int v41; // [rsp+18h] [rbp-48h]
  unsigned int v42; // [rsp+28h] [rbp-38h] BYREF
  unsigned int v43[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v8 = (__int64)a3;
  v9 = *a1;
  if ( !*(_BYTE *)(v9 + 28) )
  {
LABEL_18:
    sub_C8CC70(v9, v8, (__int64)a3, a4, a5, a6);
    v11 = a1[1];
    v12 = *(_DWORD *)(v11 + 24);
    if ( v12 )
      goto LABEL_19;
LABEL_7:
    ++*(_QWORD *)v11;
    goto LABEL_8;
  }
  v10 = *(__int64 **)(v9 + 8);
  a4 = *(unsigned int *)(v9 + 20);
  a3 = &v10[a4];
  if ( v10 == a3 )
  {
LABEL_17:
    if ( (unsigned int)a4 >= *(_DWORD *)(v9 + 16) )
      goto LABEL_18;
    *(_DWORD *)(v9 + 20) = a4 + 1;
    *a3 = v8;
    ++*(_QWORD *)v9;
  }
  else
  {
    while ( v8 != *v10 )
    {
      if ( a3 == ++v10 )
        goto LABEL_17;
    }
  }
  v11 = a1[1];
  v12 = *(_DWORD *)(v11 + 24);
  if ( !v12 )
    goto LABEL_7;
LABEL_19:
  v40 = *(_QWORD *)(v11 + 8);
  v38 = v12;
  v23 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v42 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
  v43[0] = v23;
  v24 = sub_28052C0(v43, &v42);
  v12 = v38;
  v20 = 0;
  v25 = 1;
  for ( i = (v38 - 1) & v24; ; i = (v38 - 1) & v29 )
  {
    result = (__int64 *)(v40 + 16LL * i);
    v28 = *result;
    if ( a2 == *result && v8 == result[1] )
      return result;
    if ( v28 == -4096 )
      break;
    if ( v28 == -8192 && result[1] == -8192 && !v20 )
      v20 = (__int64 *)(v40 + 16LL * i);
LABEL_26:
    v29 = v25 + i;
    ++v25;
  }
  if ( result[1] != -4096 )
    goto LABEL_26;
  if ( !v20 )
    v20 = (__int64 *)(v40 + 16LL * i);
  v30 = *(_DWORD *)(v11 + 16);
  ++*(_QWORD *)v11;
  v31 = v30 + 1;
  if ( 4 * (v30 + 1) < 3 * v38 )
  {
    result = (__int64 *)(v38 - *(_DWORD *)(v11 + 20) - v31);
    if ( (unsigned int)result > v38 >> 3 )
      goto LABEL_35;
    sub_2805990(v11, v38);
    v41 = *(_DWORD *)(v11 + 24);
    if ( v41 )
    {
      v42 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
      v32 = *(_QWORD *)(v11 + 8);
      v43[0] = v23;
      v39 = v32;
      v33 = sub_28052C0(v43, &v42);
      v20 = 0;
      v34 = 1;
      for ( j = (v41 - 1) & v33; ; j = (v41 - 1) & v37 )
      {
        result = (__int64 *)(v39 + 16LL * j);
        v36 = *result;
        if ( a2 == *result && v8 == result[1] )
        {
          v31 = *(_DWORD *)(v11 + 16) + 1;
          v20 = (__int64 *)(v39 + 16LL * j);
          goto LABEL_35;
        }
        if ( v36 == -4096 )
        {
          if ( result[1] == -4096 )
          {
            if ( !v20 )
              v20 = (__int64 *)(v39 + 16LL * j);
            v31 = *(_DWORD *)(v11 + 16) + 1;
            goto LABEL_35;
          }
        }
        else if ( v36 == -8192 && result[1] == -8192 && !v20 )
        {
          v20 = (__int64 *)(v39 + 16LL * j);
        }
        v37 = v34 + j;
        ++v34;
      }
    }
LABEL_61:
    ++*(_DWORD *)(v11 + 16);
    BUG();
  }
LABEL_8:
  sub_2805990(v11, 2 * v12);
  v13 = *(_DWORD *)(v11 + 24);
  if ( !v13 )
    goto LABEL_61;
  v14 = v13 - 1;
  v15 = *(_QWORD *)(v11 + 8);
  v42 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
  v43[0] = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v16 = sub_28052C0(v43, &v42);
  v17 = 1;
  v18 = 0;
  v19 = v14 & v16;
  while ( 2 )
  {
    v20 = (__int64 *)(v15 + 16LL * v19);
    v21 = *v20;
    if ( a2 == *v20 && v8 == v20[1] )
    {
      result = (__int64 *)*(unsigned int *)(v11 + 16);
      v31 = (_DWORD)result + 1;
      goto LABEL_35;
    }
    if ( v21 != -4096 )
    {
      if ( v21 == -8192 && v20[1] == -8192 && !v18 )
        v18 = (__int64 *)(v15 + 16LL * v19);
      goto LABEL_16;
    }
    if ( v20[1] != -4096 )
    {
LABEL_16:
      v22 = v17 + v19;
      ++v17;
      v19 = v14 & v22;
      continue;
    }
    break;
  }
  result = (__int64 *)*(unsigned int *)(v11 + 16);
  if ( v18 )
    v20 = v18;
  v31 = (_DWORD)result + 1;
LABEL_35:
  *(_DWORD *)(v11 + 16) = v31;
  if ( *v20 != -4096 || v20[1] != -4096 )
    --*(_DWORD *)(v11 + 20);
  *v20 = a2;
  v20[1] = v8;
  return result;
}
