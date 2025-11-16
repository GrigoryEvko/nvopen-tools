// Function: sub_E3C0A0
// Address: 0xe3c0a0
//
_QWORD *__fastcall sub_E3C0A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 *v6; // r12
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rax
  __int64 *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 *v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 *v20; // r15
  __int64 *v21; // r12
  __int64 v22; // rdi
  __int64 *v23; // r14
  __int64 v24; // rdx
  __int64 *v25; // r12
  __int64 *v26; // rsi
  _QWORD *result; // rax
  _QWORD *v28; // rdx
  _QWORD *v29; // rcx

  v3 = a1 + 72;
  if ( *(_QWORD *)a3 )
    v3 = *(_QWORD *)a3 + 32LL;
  v6 = *(__int64 **)v3;
  v7 = *(_QWORD *)(v3 + 8) - *(_QWORD *)v3;
  v8 = v7 >> 5;
  v9 = v7 >> 3;
  if ( v8 <= 0 )
  {
LABEL_49:
    if ( v9 != 2 )
    {
      if ( v9 != 3 )
      {
        if ( v9 != 1 )
        {
          v11 = *(__int64 **)(a2 + 40);
          v6 = *(__int64 **)(v3 + 8);
          if ( v11 != *(__int64 **)(a2 + 48) )
            goto LABEL_11;
LABEL_53:
          sub_E38EC0((__int64 **)(a2 + 32), (__int64)v11, v6);
          goto LABEL_14;
        }
LABEL_58:
        if ( a3 != *v6 )
          v6 = *(__int64 **)(v3 + 8);
        goto LABEL_10;
      }
      if ( a3 == *v6 )
        goto LABEL_10;
      ++v6;
    }
    if ( a3 == *v6 )
      goto LABEL_10;
    ++v6;
    goto LABEL_58;
  }
  v10 = &v6[4 * v8];
  while ( a3 != *v6 )
  {
    if ( a3 == v6[1] )
    {
      ++v6;
      break;
    }
    if ( a3 == v6[2] )
    {
      v6 += 2;
      break;
    }
    if ( a3 == v6[3] )
    {
      v6 += 3;
      break;
    }
    v6 += 4;
    if ( v10 == v6 )
    {
      v9 = (__int64)(*(_QWORD *)(v3 + 8) - (_QWORD)v6) >> 3;
      goto LABEL_49;
    }
  }
LABEL_10:
  v11 = *(__int64 **)(a2 + 40);
  if ( v11 == *(__int64 **)(a2 + 48) )
    goto LABEL_53;
LABEL_11:
  if ( v11 )
  {
    *v11 = *v6;
    *v6 = 0;
    v11 = *(__int64 **)(a2 + 40);
  }
  *(_QWORD *)(a2 + 40) = ++v11;
LABEL_14:
  v12 = *(_QWORD *)(v3 + 8);
  v13 = *(_QWORD *)(v12 - 8);
  *(_QWORD *)(v12 - 8) = 0;
  v14 = *v6;
  *v6 = v13;
  if ( v14 )
    sub_E38110(v14, (__int64)v11);
  v15 = (__int64 *)(*(_QWORD *)(v3 + 8) - 8LL);
  *(_QWORD *)(v3 + 8) = v15;
  v16 = *v15;
  if ( *v15 )
  {
    v17 = *(_QWORD *)(v16 + 176);
    if ( v17 != v16 + 192 )
      _libc_free(v17, v11);
    v18 = *(_QWORD *)(v16 + 88);
    if ( v18 != v16 + 104 )
      _libc_free(v18, v11);
    v19 = 8LL * *(unsigned int *)(v16 + 80);
    sub_C7D6A0(*(_QWORD *)(v16 + 64), v19, 8);
    v20 = *(__int64 **)(v16 + 40);
    v21 = *(__int64 **)(v16 + 32);
    if ( v20 != v21 )
    {
      do
      {
        if ( *v21 )
          sub_E38110(*v21, v19);
        ++v21;
      }
      while ( v20 != v21 );
      v21 = *(__int64 **)(v16 + 32);
    }
    if ( v21 )
    {
      v19 = *(_QWORD *)(v16 + 48) - (_QWORD)v21;
      j_j___libc_free_0(v21, v19);
    }
    v22 = *(_QWORD *)(v16 + 8);
    if ( v22 != v16 + 24 )
      _libc_free(v22, v19);
    j_j___libc_free_0(v16, 224);
  }
  v23 = *(__int64 **)(a3 + 88);
  v24 = *(unsigned int *)(a3 + 96);
  *(_QWORD *)a3 = a2;
  v25 = &v23[v24];
  while ( v23 != v25 )
  {
    v26 = v23++;
    sub_E3B670(a2 + 56, v26);
  }
  result = (_QWORD *)*(unsigned int *)(a1 + 56);
  if ( (_DWORD)result )
  {
    result = (_QWORD *)a1;
    v28 = *(_QWORD **)(a1 + 48);
    v29 = &v28[2 * *(unsigned int *)(a1 + 64)];
    if ( v28 != v29 )
    {
      while ( 1 )
      {
        result = v28;
        if ( *v28 != -4096 && *v28 != -8192 )
          break;
        v28 += 2;
        if ( v29 == v28 )
          goto LABEL_34;
      }
      while ( v29 != result )
      {
        if ( result[1] == a3 )
          result[1] = a2;
        result += 2;
        if ( result == v29 )
          break;
        while ( *result == -8192 || *result == -4096 )
        {
          result += 2;
          if ( v29 == result )
            goto LABEL_34;
        }
      }
    }
  }
LABEL_34:
  *(_DWORD *)(a2 + 184) = 0;
  *(_DWORD *)(a3 + 184) = 0;
  return result;
}
