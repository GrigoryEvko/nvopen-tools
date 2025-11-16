// Function: sub_325FB70
// Address: 0x325fb70
//
size_t __fastcall sub_325FB70(__int64 a1, char *a2, char *a3, char *a4)
{
  char *v5; // r13
  __int64 v6; // r9
  char *v8; // r12
  __int64 v10; // r11
  __int64 v11; // rcx
  size_t v12; // rsi
  unsigned __int64 v13; // rdi
  size_t result; // rax
  char *v15; // r10
  unsigned __int64 v16; // rdx
  char *v17; // r8
  size_t v18; // rdx
  __int64 v19; // r15
  unsigned int v20; // ecx
  size_t v21; // r11
  char *v22; // r15
  unsigned __int64 v23; // rdx
  void *v24; // rdi
  __int64 v25; // [rsp+0h] [rbp-60h]
  __int64 v26; // [rsp+8h] [rbp-58h]
  size_t v27; // [rsp+8h] [rbp-58h]
  size_t v28; // [rsp+10h] [rbp-50h]
  size_t v29; // [rsp+10h] [rbp-50h]
  int v30; // [rsp+18h] [rbp-48h]
  char *v31; // [rsp+18h] [rbp-48h]
  char *v32; // [rsp+18h] [rbp-48h]
  char *v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  char *v35; // [rsp+20h] [rbp-40h]
  int v36; // [rsp+20h] [rbp-40h]
  char *v37; // [rsp+20h] [rbp-40h]
  size_t v38; // [rsp+28h] [rbp-38h]
  int v39; // [rsp+28h] [rbp-38h]
  char *v40; // [rsp+28h] [rbp-38h]
  __int64 v41; // [rsp+28h] [rbp-38h]
  __int64 v42; // [rsp+28h] [rbp-38h]
  __int64 v43; // [rsp+28h] [rbp-38h]

  v5 = a2;
  v6 = a4 - a3;
  v8 = a3;
  v10 = (a4 - a3) >> 2;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v13 = *(unsigned int *)(a1 + 12);
  result = 4 * v11;
  v15 = &a2[-v12];
  v16 = v11 + v10;
  v17 = (char *)(v12 + 4 * v11);
  if ( v5 == v17 )
  {
    if ( v16 > v13 )
    {
      v36 = v10;
      v43 = v6;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 4u, (__int64)v17, v6);
      v11 = *(unsigned int *)(a1 + 8);
      result = *(_QWORD *)a1;
      LODWORD(v10) = v36;
      v6 = v43;
      v17 = (char *)(*(_QWORD *)a1 + 4 * v11);
    }
    if ( v8 != a4 )
    {
      v39 = v10;
      result = (size_t)memcpy(v17, v8, v6);
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
      LODWORD(v10) = v39;
    }
    *(_DWORD *)(a1 + 8) = v10 + v11;
  }
  else
  {
    if ( v16 > v13 )
    {
      v30 = v10;
      v34 = v6;
      v40 = v15;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 4u, (__int64)v17, v6);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_QWORD *)a1;
      v15 = v40;
      LODWORD(v10) = v30;
      result = 4 * v11;
      v6 = v34;
      v5 = &v40[*(_QWORD *)a1];
      v17 = (char *)(*(_QWORD *)a1 + 4 * v11);
    }
    v18 = result - (_QWORD)v15;
    v19 = (__int64)(result - (_QWORD)v15) >> 2;
    if ( result - (unsigned __int64)v15 >= v6 )
    {
      v21 = result - v6;
      result = v6;
      v22 = (char *)(v12 + v21);
      v23 = v11 + (v6 >> 2);
      v41 = v6 >> 2;
      v24 = v17;
      if ( v23 > *(unsigned int *)(a1 + 12) )
      {
        v25 = v6;
        v27 = v6;
        v29 = v21;
        v32 = v17;
        v37 = v15;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v23, 4u, (__int64)v17, v6);
        v11 = *(unsigned int *)(a1 + 8);
        v6 = v25;
        result = v27;
        v21 = v29;
        v17 = v32;
        v24 = (void *)(*(_QWORD *)a1 + 4 * v11);
        v15 = v37;
      }
      if ( v22 != v17 )
      {
        v26 = v6;
        v28 = v21;
        v31 = v17;
        v35 = v15;
        result = (size_t)memmove(v24, v22, result);
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
        v6 = v26;
        v21 = v28;
        v17 = v31;
        v15 = v35;
      }
      *(_DWORD *)(a1 + 8) = v41 + v11;
      if ( v22 != v5 )
      {
        v42 = v6;
        result = (size_t)memmove(&v17[-(v21 - (_QWORD)v15)], v5, v21 - (_QWORD)v15);
        v6 = v42;
      }
      if ( v8 != a4 )
        return (size_t)memmove(v5, v8, v6);
    }
    else
    {
      v20 = v10 + v11;
      *(_DWORD *)(a1 + 8) = v20;
      if ( v17 != v5 )
      {
        v33 = v17;
        v38 = result - (_QWORD)v15;
        result = (size_t)memcpy((void *)(v12 + 4LL * v20 - v18), v5, v18);
        v17 = v33;
        v18 = v38;
      }
      if ( v19 )
      {
        for ( result = 0; result != v19; ++result )
          *(_DWORD *)&v5[4 * result] = *(_DWORD *)&v8[4 * result];
        v8 += v18;
      }
      if ( a4 != v8 )
        return (size_t)memcpy(v17, v8, a4 - v8);
    }
  }
  return result;
}
