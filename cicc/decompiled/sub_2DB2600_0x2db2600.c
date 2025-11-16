// Function: sub_2DB2600
// Address: 0x2db2600
//
__int64 *__fastcall sub_2DB2600(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v7; // rsi
  unsigned int v8; // eax
  unsigned int v9; // ecx
  __int64 v10; // r15
  __int64 *result; // rax
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rdx
  unsigned int i; // eax
  __int64 v16; // r13
  __int64 v17; // r8
  __int64 v18; // r14
  __int64 v19; // r12
  _QWORD *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rsi
  _QWORD *v23; // rax
  __int64 v24; // r9
  int v25; // r11d
  __int64 v26; // r8
  size_t v27; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r10
  _QWORD *v34; // r14
  __int64 *v35; // rax
  __int64 v36; // r11
  __int64 *v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r10
  __int64 v40; // r8
  unsigned __int64 v41; // r14
  unsigned __int64 v42; // rdi
  __int64 *v43; // [rsp+8h] [rbp-68h]
  __int64 v44; // [rsp+18h] [rbp-58h]
  __int64 *v45; // [rsp+20h] [rbp-50h]
  __int64 v46[7]; // [rsp+38h] [rbp-38h] BYREF

  if ( a2 )
  {
    v7 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
    v8 = v7;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  v9 = *(_DWORD *)(a1 + 32);
  v10 = 0;
  if ( v8 < v9 )
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v7);
  result = &a3[a4];
  v43 = result;
  if ( a3 != result )
  {
    v45 = a3;
    v12 = *a3;
    v44 = *a3;
    if ( !*a3 )
      goto LABEL_28;
LABEL_7:
    v13 = *(_DWORD *)(v12 + 24);
    v14 = (unsigned int)(v13 + 1);
    for ( i = v13 + 1; ; i = 0 )
    {
      v16 = 0;
      if ( i < v9 )
        v16 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v14);
      while ( 1 )
      {
        v17 = *(unsigned int *)(v16 + 32);
        if ( !*(_DWORD *)(v16 + 32) )
          break;
        while ( 1 )
        {
          v18 = *(_QWORD *)(*(_QWORD *)(v16 + 24) + 8 * v17 - 8);
          *(_BYTE *)(a1 + 112) = 0;
          v19 = *(_QWORD *)(v18 + 8);
          if ( v10 == v19 )
            break;
          v20 = *(_QWORD **)(v19 + 24);
          v21 = *(unsigned int *)(v19 + 32);
          v46[0] = v18;
          v22 = (__int64)&v20[v21];
          v23 = sub_2DB1E30(v20, v22, v46);
          v26 = (__int64)(v23 + 1);
          if ( v23 + 1 != (_QWORD *)v22 )
          {
            v27 = v22 - v26;
            v22 = (__int64)(v23 + 1);
            memmove(v23, v23 + 1, v27);
            v25 = *(_DWORD *)(v19 + 32);
          }
          *(_DWORD *)(v19 + 32) = v25 - 1;
          *(_QWORD *)(v18 + 8) = v10;
          v28 = *(unsigned int *)(v10 + 32);
          v29 = *(unsigned int *)(v10 + 36);
          if ( v28 + 1 > v29 )
          {
            v22 = v10 + 40;
            sub_C8D5F0(v10 + 24, (const void *)(v10 + 40), v28 + 1, 8u, v26, v24);
            v28 = *(unsigned int *)(v10 + 32);
          }
          v30 = *(_QWORD *)(v10 + 24);
          *(_QWORD *)(v30 + 8 * v28) = v18;
          ++*(_DWORD *)(v10 + 32);
          if ( *(_DWORD *)(v18 + 16) == *(_DWORD *)(*(_QWORD *)(v18 + 8) + 16LL) + 1 )
            break;
          sub_2DB2200(v18, v22, v30, v29, v26, v24);
          v17 = *(unsigned int *)(v16 + 32);
          if ( !*(_DWORD *)(v16 + 32) )
            goto LABEL_18;
        }
      }
LABEL_18:
      if ( v44 )
        v17 = 8LL * (unsigned int)(*(_DWORD *)(v44 + 24) + 1);
      v31 = (__int64 *)(v17 + *(_QWORD *)(a1 + 24));
      v32 = *v31;
      *(_BYTE *)(a1 + 112) = 0;
      v33 = *(_QWORD *)(v32 + 8);
      v46[0] = v32;
      if ( v33 )
      {
        v34 = *(_QWORD **)(v33 + 24);
        v35 = sub_2DB1E30(v34, (__int64)&v34[*(unsigned int *)(v33 + 32)], v46);
        v37 = (_QWORD *)((char *)v34 + v36 - 8);
        v38 = *v35;
        *v35 = *v37;
        *v37 = v38;
        --*(_DWORD *)(v39 + 32);
        v31 = (__int64 *)(v40 + *(_QWORD *)(a1 + 24));
      }
      v41 = *v31;
      *v31 = 0;
      if ( v41 )
      {
        v42 = *(_QWORD *)(v41 + 24);
        if ( v42 != v41 + 40 )
          _libc_free(v42);
        j_j___libc_free_0(v41);
      }
      result = ++v45;
      if ( v43 == v45 )
        break;
      v9 = *(_DWORD *)(a1 + 32);
      v12 = *v45;
      v44 = *v45;
      if ( *v45 )
        goto LABEL_7;
LABEL_28:
      v14 = 0;
    }
  }
  return result;
}
