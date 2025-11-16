// Function: sub_9D3840
// Address: 0x9d3840
//
__int64 *__fastcall sub_9D3840(__int64 *a1, char *a2, __int64 *a3)
{
  __int64 *v3; // r14
  __int64 v5; // rbx
  char *v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rsi
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // r15
  _QWORD *v14; // r9
  __int64 v15; // rsi
  int v16; // r10d
  int v17; // r8d
  __int64 v18; // r15
  __int64 *v19; // rcx
  __int64 v20; // rsi
  int v21; // eax
  __int64 v22; // rcx
  int v23; // edx
  char *i; // r13
  char *v25; // rdi
  char *v26; // rdi
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 *v30; // [rsp+8h] [rbp-58h]
  __int64 *v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 *v34; // [rsp+20h] [rbp-40h]
  __int64 *v35; // [rsp+20h] [rbp-40h]
  _QWORD *v36; // [rsp+20h] [rbp-40h]
  __int64 v37; // [rsp+28h] [rbp-38h]

  v3 = (__int64 *)a2;
  v5 = a1[1];
  v6 = (char *)*a1;
  v7 = 0xF0F0F0F0F0F0F0F1LL * ((v5 - *a1) >> 3);
  if ( v7 == 0xF0F0F0F0F0F0F0LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xF0F0F0F0F0F0F0F1LL * ((v5 - (__int64)v6) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0xF0F0F0F0F0F0F0FLL * ((v5 - (__int64)v6) >> 3);
  v11 = a2 - v6;
  v12 = v9;
  if ( v9 )
  {
    v28 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v10 )
    {
      v32 = 0;
      v13 = 136;
      v37 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0xF0F0F0F0F0F0F0LL )
      v10 = 0xF0F0F0F0F0F0F0LL;
    v28 = 136 * v10;
  }
  v30 = a3;
  v29 = sub_22077B0(v28);
  v11 = a2 - v6;
  a3 = v30;
  v37 = v29;
  v32 = v29 + v28;
  v13 = v29 + 136;
LABEL_7:
  v14 = (_QWORD *)(v37 + v11);
  if ( v14 )
  {
    v15 = *a3;
    v16 = *((_DWORD *)a3 + 4);
    v14[2] = 0xC00000000LL;
    *v14 = v15;
    v14[1] = v14 + 3;
    if ( v16 )
    {
      v31 = a3;
      v36 = v14;
      sub_9C31C0((__int64)(v14 + 1), (char **)a3 + 1);
      a3 = v31;
      v14 = v36;
    }
    v17 = *((_DWORD *)a3 + 20);
    v12 = (__int64)(v14 + 11);
    v14[9] = v14 + 11;
    v14[10] = 0xC00000000LL;
    if ( v17 )
    {
      v12 = (__int64)(a3 + 9);
      sub_9C31C0((__int64)(v14 + 9), (char **)a3 + 9);
    }
  }
  if ( a2 != v6 )
  {
    v18 = v37;
    v19 = (__int64 *)v6;
    while ( 1 )
    {
      if ( v18 )
      {
        v20 = *v19;
        *(_DWORD *)(v18 + 16) = 0;
        *(_DWORD *)(v18 + 20) = 12;
        *(_QWORD *)v18 = v20;
        *(_QWORD *)(v18 + 8) = v18 + 24;
        if ( *((_DWORD *)v19 + 4) )
        {
          v34 = v19;
          sub_9C2E20(v18 + 8, (__int64)(v19 + 1));
          v19 = v34;
        }
        *(_DWORD *)(v18 + 80) = 0;
        *(_QWORD *)(v18 + 72) = v18 + 88;
        *(_DWORD *)(v18 + 84) = 12;
        if ( *((_DWORD *)v19 + 20) )
        {
          v35 = v19;
          sub_9C2E20(v18 + 72, (__int64)(v19 + 9));
          v19 = v35;
        }
      }
      v19 += 17;
      v12 = v18 + 136;
      if ( a2 == (char *)v19 )
        break;
      v18 += 136;
    }
    v13 = v18 + 272;
  }
  if ( a2 != (char *)v5 )
  {
    do
    {
      v22 = *v3;
      v23 = *((_DWORD *)v3 + 4);
      *(_DWORD *)(v13 + 16) = 0;
      *(_DWORD *)(v13 + 20) = 12;
      *(_QWORD *)v13 = v22;
      *(_QWORD *)(v13 + 8) = v13 + 24;
      if ( v23 )
      {
        v12 = (__int64)(v3 + 1);
        sub_9C2E20(v13 + 8, (__int64)(v3 + 1));
      }
      v21 = *((_DWORD *)v3 + 20);
      *(_DWORD *)(v13 + 80) = 0;
      *(_QWORD *)(v13 + 72) = v13 + 88;
      *(_DWORD *)(v13 + 84) = 12;
      if ( v21 )
      {
        v12 = (__int64)(v3 + 9);
        sub_9C2E20(v13 + 72, (__int64)(v3 + 9));
      }
      v3 += 17;
      v13 += 136;
    }
    while ( (__int64 *)v5 != v3 );
  }
  for ( i = v6; i != (char *)v5; i += 136 )
  {
    v25 = (char *)*((_QWORD *)i + 9);
    if ( v25 != i + 88 )
      _libc_free(v25, v12);
    v26 = (char *)*((_QWORD *)i + 1);
    if ( v26 != i + 24 )
      _libc_free(v26, v12);
  }
  if ( v6 )
    j_j___libc_free_0(v6, a1[2] - (_QWORD)v6);
  *a1 = v37;
  a1[1] = v13;
  a1[2] = v32;
  return a1;
}
