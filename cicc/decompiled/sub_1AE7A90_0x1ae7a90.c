// Function: sub_1AE7A90
// Address: 0x1ae7a90
//
void __fastcall sub_1AE7A90(__int64 a1, char *a2, size_t a3, unsigned __int8 *a4, __int64 a5, int a6)
{
  char *v8; // r13
  __int64 v11; // rax
  __int64 v12; // rsi
  size_t v13; // rcx
  char *v14; // r15
  char *v15; // r8
  unsigned __int64 v16; // r9
  __int64 v17; // rdx
  size_t v18; // rcx
  unsigned int v19; // edx
  size_t v20; // r10
  void *v21; // rdi
  size_t v22; // r9
  size_t v23; // r9
  size_t v24; // [rsp+0h] [rbp-50h]
  char *v25; // [rsp+8h] [rbp-48h]
  char *v26; // [rsp+8h] [rbp-48h]
  size_t v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  size_t v29; // [rsp+18h] [rbp-38h]
  size_t v30; // [rsp+18h] [rbp-38h]
  char *v31; // [rsp+18h] [rbp-38h]
  int v32; // [rsp+18h] [rbp-38h]

  v8 = a2;
  v11 = *(_QWORD *)a1;
  v12 = *(unsigned int *)(a1 + 8);
  v13 = *(unsigned int *)(a1 + 12);
  v14 = (char *)(*(_QWORD *)a1 + v12);
  v15 = &a2[-*(_QWORD *)a1];
  if ( v8 == v14 )
  {
    if ( a3 > v13 - v12 )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), v12 + a3, 1, (int)v15, a6);
      v12 = *(unsigned int *)(a1 + 8);
      v14 = (char *)(v12 + *(_QWORD *)a1);
    }
    if ( a3 )
    {
      memset(v14, *a4, a3);
      LODWORD(v12) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = a3 + v12;
  }
  else
  {
    v16 = v12 + a3;
    v17 = *(unsigned int *)(a1 + 8);
    if ( v12 + a3 > v13 )
    {
      v31 = v15;
      sub_16CD150(a1, (const void *)(a1 + 16), v16, 1, (int)v15, v16);
      v17 = *(unsigned int *)(a1 + 8);
      v15 = v31;
      v11 = *(_QWORD *)a1;
      LODWORD(v12) = *(_DWORD *)(a1 + 8);
      v18 = v17 - (_QWORD)v31;
      v8 = &v31[*(_QWORD *)a1];
      v14 = (char *)(*(_QWORD *)a1 + v17);
      if ( a3 > v17 - (__int64)v31 )
        goto LABEL_4;
    }
    else
    {
      v18 = v12 - (_QWORD)v15;
      if ( a3 > v12 - (__int64)v15 )
      {
LABEL_4:
        v19 = a3 + v17;
        *(_DWORD *)(a1 + 8) = v19;
        if ( v8 != v14 )
        {
          v29 = v18;
          memcpy((void *)(v11 + v19 - v18), v8, v18);
          v18 = v29;
        }
        if ( v18 )
        {
          v30 = v18;
          memset(v8, *a4, v18);
          v18 = v30;
        }
        memset(v14, *a4, a3 - v18);
        return;
      }
    }
    v20 = a3;
    v21 = v14;
    v22 = v17 - a3;
    if ( a3 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v17 )
    {
      v24 = v17 - a3;
      v26 = v15;
      v28 = v11;
      sub_16CD150(a1, (const void *)(a1 + 16), a3 + v17, 1, (int)v15, v22);
      v12 = *(unsigned int *)(a1 + 8);
      v22 = v24;
      v15 = v26;
      v11 = v28;
      v20 = a3;
      v21 = (void *)(v12 + *(_QWORD *)a1);
    }
    if ( v20 )
    {
      v25 = v15;
      v27 = v22;
      v32 = v20;
      memmove(v21, (const void *)(v11 + v22), v20);
      LODWORD(v12) = *(_DWORD *)(a1 + 8);
      v15 = v25;
      v22 = v27;
      LODWORD(v20) = v32;
    }
    *(_DWORD *)(a1 + 8) = v20 + v12;
    v23 = v22 - (_QWORD)v15;
    if ( v23 )
      memmove(&v14[-v23], v8, v23);
    if ( a3 )
      memset(v8, *a4, a3);
  }
}
