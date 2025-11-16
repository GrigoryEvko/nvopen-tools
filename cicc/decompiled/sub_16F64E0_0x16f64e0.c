// Function: sub_16F64E0
// Address: 0x16f64e0
//
void __fastcall sub_16F64E0(__int64 a1, char *a2, char *a3, char *a4)
{
  char *v7; // r13
  size_t v8; // r9
  char *v9; // r12
  __int64 v10; // rax
  __int64 v11; // rcx
  size_t v12; // rsi
  char *v13; // r8
  char *v14; // r10
  __int64 v15; // rdx
  size_t v16; // rbx
  unsigned int v17; // edx
  __int64 i; // rax
  size_t v19; // rbx
  void *v20; // rdi
  size_t v21; // r15
  size_t v22; // r15
  size_t v23; // [rsp+0h] [rbp-50h]
  size_t v24; // [rsp+8h] [rbp-48h]
  char *v25; // [rsp+8h] [rbp-48h]
  size_t v26; // [rsp+10h] [rbp-40h]
  char *v27; // [rsp+10h] [rbp-40h]
  char *v28; // [rsp+10h] [rbp-40h]
  char *v29; // [rsp+18h] [rbp-38h]
  int v30; // [rsp+18h] [rbp-38h]
  char *v31; // [rsp+18h] [rbp-38h]
  size_t v32; // [rsp+18h] [rbp-38h]
  size_t v33; // [rsp+18h] [rbp-38h]
  char *v34; // [rsp+18h] [rbp-38h]
  __int64 v35; // [rsp+18h] [rbp-38h]

  v7 = a2;
  v8 = a4 - a3;
  v9 = a3;
  v10 = *(_QWORD *)a1;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(unsigned int *)(a1 + 12);
  v13 = (char *)(*(_QWORD *)a1 + v11);
  v14 = &a2[-*(_QWORD *)a1];
  if ( v7 == v13 )
  {
    if ( v12 - v11 < v8 )
    {
      v32 = v8;
      sub_16CD150(a1, (const void *)(a1 + 16), v11 + v8, 1, (int)v13, v8);
      v11 = *(unsigned int *)(a1 + 8);
      v8 = v32;
      v13 = (char *)(v11 + *(_QWORD *)a1);
    }
    if ( v9 != a4 )
    {
      v30 = v8;
      memcpy(v13, v9, v8);
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
      LODWORD(v8) = v30;
    }
    *(_DWORD *)(a1 + 8) = v8 + v11;
  }
  else
  {
    v15 = *(unsigned int *)(a1 + 8);
    if ( v11 + v8 > v12 )
    {
      v26 = v8;
      v31 = v14;
      sub_16CD150(a1, (const void *)(a1 + 16), v11 + v8, 1, (int)v13, v8);
      v15 = *(unsigned int *)(a1 + 8);
      v14 = v31;
      v10 = *(_QWORD *)a1;
      v8 = v26;
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
      v16 = v15 - (_QWORD)v31;
      v7 = &v31[*(_QWORD *)a1];
      v13 = (char *)(*(_QWORD *)a1 + v15);
      if ( v15 - (__int64)v31 < v26 )
        goto LABEL_4;
    }
    else
    {
      v16 = v11 - (_QWORD)v14;
      if ( v11 - (__int64)v14 < v8 )
      {
LABEL_4:
        v17 = v8 + v15;
        *(_DWORD *)(a1 + 8) = v17;
        if ( v13 != v7 )
        {
          v29 = v13;
          memcpy((void *)(v10 + v17 - v16), v7, v16);
          v13 = v29;
        }
        if ( v16 )
        {
          for ( i = 0; i != v16; ++i )
            v7[i] = v9[i];
          v9 += v16;
        }
        if ( a4 != v9 )
          memcpy(v13, v9, a4 - v9);
        return;
      }
    }
    v19 = v8;
    v20 = v13;
    v21 = v15 - v8;
    if ( v8 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v15 )
    {
      v23 = v8;
      v25 = v13;
      v28 = v14;
      v35 = v10;
      sub_16CD150(a1, (const void *)(a1 + 16), v8 + v15, 1, (int)v13, v8);
      v11 = *(unsigned int *)(a1 + 8);
      v8 = v23;
      v13 = v25;
      v14 = v28;
      v10 = v35;
      v20 = (void *)(v11 + *(_QWORD *)a1);
    }
    if ( v19 )
    {
      v24 = v8;
      v27 = v13;
      v34 = v14;
      memmove(v20, (const void *)(v10 + v21), v19);
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
      v8 = v24;
      v13 = v27;
      v14 = v34;
    }
    *(_DWORD *)(a1 + 8) = v19 + v11;
    v22 = v21 - (_QWORD)v14;
    if ( v22 )
    {
      v33 = v8;
      memmove(&v13[-v22], v7, v22);
      v8 = v33;
    }
    if ( v8 )
      memmove(v7, v9, v8);
  }
}
