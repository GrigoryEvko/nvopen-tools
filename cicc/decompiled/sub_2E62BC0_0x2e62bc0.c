// Function: sub_2E62BC0
// Address: 0x2e62bc0
//
_QWORD *__fastcall sub_2E62BC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 *v6; // r12
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rax
  unsigned __int64 *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // rdx
  __int64 *v20; // r14
  __int64 v21; // rdx
  __int64 *v22; // r12
  __int64 *v23; // rsi
  _QWORD *result; // rax
  _QWORD *v25; // rdx
  _QWORD *v26; // rcx
  unsigned __int64 *v28; // [rsp+8h] [rbp-38h]

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
          v11 = *(unsigned __int64 **)(a2 + 40);
          v6 = *(__int64 **)(v3 + 8);
          if ( v11 != *(unsigned __int64 **)(a2 + 48) )
            goto LABEL_11;
LABEL_53:
          sub_2E5F820((unsigned __int64 *)(a2 + 32), v11, v6);
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
  v11 = *(unsigned __int64 **)(a2 + 40);
  if ( v11 == *(unsigned __int64 **)(a2 + 48) )
    goto LABEL_53;
LABEL_11:
  if ( v11 )
  {
    *v11 = *v6;
    *v6 = 0;
    v11 = *(unsigned __int64 **)(a2 + 40);
  }
  *(_QWORD *)(a2 + 40) = v11 + 1;
LABEL_14:
  v12 = *(_QWORD *)(v3 + 8);
  v13 = *(_QWORD *)(v12 - 8);
  *(_QWORD *)(v12 - 8) = 0;
  v14 = *v6;
  *v6 = v13;
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 176);
    if ( v15 != v14 + 192 )
      _libc_free(v15);
    v16 = *(_QWORD *)(v14 + 88);
    if ( v16 != v14 + 104 )
      _libc_free(v16);
    sub_C7D6A0(*(_QWORD *)(v14 + 64), 8LL * *(unsigned int *)(v14 + 80), 8);
    v17 = *(unsigned __int64 **)(v14 + 32);
    v28 = *(unsigned __int64 **)(v14 + 40);
    if ( v28 != v17 )
    {
      do
      {
        if ( *v17 )
          sub_2E5DCD0(*v17);
        ++v17;
      }
      while ( v28 != v17 );
      v17 = *(unsigned __int64 **)(v14 + 32);
    }
    if ( v17 )
      j_j___libc_free_0((unsigned __int64)v17);
    v18 = *(_QWORD *)(v14 + 8);
    if ( v18 != v14 + 24 )
      _libc_free(v18);
    j_j___libc_free_0(v14);
  }
  v19 = (unsigned __int64 *)(*(_QWORD *)(v3 + 8) - 8LL);
  *(_QWORD *)(v3 + 8) = v19;
  if ( *v19 )
    sub_2E5DCD0(*v19);
  v20 = *(__int64 **)(a3 + 88);
  v21 = *(unsigned int *)(a3 + 96);
  *(_QWORD *)a3 = a2;
  v22 = &v20[v21];
  while ( v20 != v22 )
  {
    v23 = v20++;
    sub_2E62120(a2 + 56, v23);
  }
  result = (_QWORD *)*(unsigned int *)(a1 + 56);
  if ( (_DWORD)result )
  {
    result = (_QWORD *)a1;
    v25 = *(_QWORD **)(a1 + 48);
    v26 = &v25[2 * *(unsigned int *)(a1 + 64)];
    if ( v25 != v26 )
    {
      while ( 1 )
      {
        result = v25;
        if ( *v25 != -4096 && *v25 != -8192 )
          break;
        v25 += 2;
        if ( v26 == v25 )
          goto LABEL_34;
      }
      while ( v26 != result )
      {
        if ( result[1] == a3 )
          result[1] = a2;
        result += 2;
        if ( result == v26 )
          break;
        while ( *result == -8192 || *result == -4096 )
        {
          result += 2;
          if ( v26 == result )
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
