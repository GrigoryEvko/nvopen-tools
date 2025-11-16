// Function: sub_25DC650
// Address: 0x25dc650
//
size_t __fastcall sub_25DC650(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  char *v7; // r14
  __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  __int64 v11; // r9
  unsigned __int64 v12; // rdi
  __int64 v13; // rsi
  char *v14; // r10
  __int64 v15; // r12
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r11
  unsigned __int64 v20; // rcx
  unsigned int v21; // edx
  size_t result; // rax
  __int64 v23; // rax
  __int64 v24; // r14
  unsigned __int64 v25; // rdx
  __int64 v26; // r11
  char *v27; // r9
  __int64 v28; // rdi
  unsigned __int64 v29; // rsi
  void *v30; // rdi
  __int64 v31; // [rsp+0h] [rbp-60h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  size_t v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  char *v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  char *v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  unsigned __int64 v39; // [rsp+20h] [rbp-40h]
  char *v40; // [rsp+20h] [rbp-40h]
  char *v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+28h] [rbp-38h]
  char *v43; // [rsp+28h] [rbp-38h]
  int v44; // [rsp+28h] [rbp-38h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  v4 = a3;
  v7 = a2;
  v9 = a3;
  v10 = *(unsigned int *)(a1 + 8);
  v11 = *(_QWORD *)a1;
  v12 = *(unsigned int *)(a1 + 12);
  v13 = 8 * v10;
  v14 = &a2[-v11];
  v15 = v11 + 8 * v10;
  if ( v7 == (char *)v15 )
  {
    if ( v4 == a4 )
    {
      result = v10;
      if ( v10 > v12 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v10, 8u, v4, v11);
        result = *(unsigned int *)(a1 + 8);
      }
    }
    else
    {
      v23 = v4;
      v24 = 0;
      do
      {
        v23 = *(_QWORD *)(v23 + 8);
        ++v24;
      }
      while ( v23 != a4 );
      v25 = v24 + v10;
      if ( v25 > v12 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v25, 8u, v4, v11);
        v15 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
      }
      do
      {
        v15 += 8;
        *(_QWORD *)(v15 - 8) = *(_QWORD *)(v9 + 24);
        v9 = *(_QWORD *)(v9 + 8);
      }
      while ( v9 != a4 );
      result = (unsigned int)(v24 + *(_DWORD *)(a1 + 8));
    }
    *(_DWORD *)(a1 + 8) = result;
  }
  else
  {
    if ( v4 == a4 )
    {
      v19 = v10;
      v18 = 0;
    }
    else
    {
      v16 = v4;
      v17 = 0;
      do
      {
        v16 = *(_QWORD *)(v16 + 8);
        ++v17;
      }
      while ( v16 != a4 );
      v18 = v17;
      v19 = v10 + v17;
    }
    if ( v19 > v12 )
    {
      v36 = v4;
      v39 = v18;
      v43 = v14;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v19, 8u, v4, v11);
      v10 = *(unsigned int *)(a1 + 8);
      v11 = *(_QWORD *)a1;
      v14 = v43;
      v4 = v36;
      v13 = 8 * v10;
      v18 = v39;
      v7 = &v43[*(_QWORD *)a1];
      v15 = *(_QWORD *)a1 + 8 * v10;
    }
    v20 = (v13 - (__int64)v14) >> 3;
    if ( v20 >= v18 )
    {
      v26 = 8 * (v10 - v18);
      result = v13 - v26;
      v27 = (char *)(v26 + v11);
      v28 = (v13 - v26) >> 3;
      v29 = v28 + v10;
      v44 = v28;
      v30 = (void *)v15;
      if ( v29 > *(unsigned int *)(a1 + 12) )
      {
        v31 = v4;
        v33 = result;
        v35 = v27;
        v38 = v26;
        v41 = v14;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v29, 8u, v4, (__int64)v27);
        v10 = *(unsigned int *)(a1 + 8);
        v4 = v31;
        result = v33;
        v27 = v35;
        v26 = v38;
        v30 = (void *)(*(_QWORD *)a1 + 8 * v10);
        v14 = v41;
      }
      if ( v27 != (char *)v15 )
      {
        v32 = v4;
        v34 = v26;
        v37 = v14;
        v40 = v27;
        result = (size_t)memmove(v30, v27, result);
        LODWORD(v10) = *(_DWORD *)(a1 + 8);
        v4 = v32;
        v26 = v34;
        v14 = v37;
        v27 = v40;
      }
      *(_DWORD *)(a1 + 8) = v44 + v10;
      if ( v27 != v7 )
      {
        v45 = v4;
        result = (size_t)memmove((void *)(v15 - (v26 - (_QWORD)v14)), v7, v26 - (_QWORD)v14);
        v4 = v45;
      }
      if ( v4 != a4 )
      {
        do
        {
          result = *(_QWORD *)(v9 + 24);
          v7 += 8;
          *((_QWORD *)v7 - 1) = result;
          v9 = *(_QWORD *)(v9 + 8);
        }
        while ( v9 != a4 );
      }
    }
    else
    {
      v21 = v18 + v10;
      *(_DWORD *)(a1 + 8) = v21;
      if ( v7 != (char *)v15 )
      {
        v42 = (v13 - (__int64)v14) >> 3;
        memcpy((void *)(v11 + 8LL * v21 - (v13 - (_QWORD)v14)), v7, v13 - (_QWORD)v14);
        v20 = v42;
      }
      result = 0;
      if ( !v20 )
        goto LABEL_15;
      do
      {
        *(_QWORD *)&v7[8 * result++] = *(_QWORD *)(v9 + 24);
        v9 = *(_QWORD *)(v9 + 8);
      }
      while ( v20 != result );
      while ( v9 != a4 )
      {
        result = *(_QWORD *)(v9 + 24);
        v15 += 8;
        *(_QWORD *)(v15 - 8) = result;
        v9 = *(_QWORD *)(v9 + 8);
LABEL_15:
        ;
      }
    }
  }
  return result;
}
