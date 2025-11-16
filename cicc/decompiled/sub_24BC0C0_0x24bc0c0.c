// Function: sub_24BC0C0
// Address: 0x24bc0c0
//
char *__fastcall sub_24BC0C0(__int64 a1, char *a2, char *a3, char *a4)
{
  __int64 v5; // r9
  __int64 v6; // r11
  char *v8; // r13
  char *v9; // r12
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  char *v14; // r10
  unsigned __int64 v15; // rdx
  char *result; // rax
  char *v17; // r8
  size_t v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r15
  unsigned int v21; // esi
  char *v22; // r11
  char *v23; // r15
  unsigned __int64 v24; // rdx
  void *v25; // rdi
  __int64 v26; // [rsp+0h] [rbp-60h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  char *v28; // [rsp+8h] [rbp-58h]
  char *v29; // [rsp+10h] [rbp-50h]
  char *v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  int v32; // [rsp+18h] [rbp-48h]
  char *v33; // [rsp+18h] [rbp-48h]
  char *v34; // [rsp+18h] [rbp-48h]
  char *v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+20h] [rbp-40h]
  char *v37; // [rsp+20h] [rbp-40h]
  int v38; // [rsp+20h] [rbp-40h]
  char *v39; // [rsp+20h] [rbp-40h]
  char *v40; // [rsp+28h] [rbp-38h]
  int v41; // [rsp+28h] [rbp-38h]
  char *v42; // [rsp+28h] [rbp-38h]
  __int64 v43; // [rsp+28h] [rbp-38h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  v5 = a4 - a3;
  v6 = (a4 - a3) >> 4;
  v8 = a2;
  v9 = a3;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v13 = *(unsigned int *)(a1 + 12);
  v14 = &a2[-v12];
  v15 = v11 + v6;
  result = (char *)(16 * v11);
  v17 = (char *)(v12 + 16 * v11);
  if ( v8 == v17 )
  {
    if ( v15 > v13 )
    {
      v38 = v6;
      v45 = v5;
      result = (char *)sub_C8D5F0(a1, (const void *)(a1 + 16), v15, 0x10u, (__int64)v17, v5);
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
      LODWORD(v6) = v38;
      v5 = v45;
      v17 = (char *)(*(_QWORD *)a1 + 16LL * (unsigned int)v11);
    }
    if ( v9 != a4 )
    {
      v41 = v6;
      result = (char *)memcpy(v17, v9, v5);
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
      LODWORD(v6) = v41;
    }
    *(_DWORD *)(a1 + 8) = v6 + v11;
  }
  else
  {
    if ( v15 > v13 )
    {
      v32 = v6;
      v36 = v5;
      v42 = v14;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v15, 0x10u, (__int64)v17, v5);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_QWORD *)a1;
      v14 = v42;
      LODWORD(v6) = v32;
      v5 = v36;
      result = (char *)(16 * v11);
      v8 = &v42[*(_QWORD *)a1];
      v17 = (char *)(*(_QWORD *)a1 + 16 * v11);
    }
    v18 = result - v14;
    v19 = (result - v14) >> 4;
    v20 = v19;
    if ( result - v14 >= (unsigned __int64)v5 )
    {
      v22 = &result[-v5];
      result = (char *)v5;
      v23 = &v22[v12];
      v24 = v11 + (v5 >> 4);
      v43 = v5 >> 4;
      v25 = v17;
      if ( v24 > *(unsigned int *)(a1 + 12) )
      {
        v26 = v5;
        v28 = (char *)v5;
        v30 = v22;
        v34 = v17;
        v39 = v14;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v24, 0x10u, (__int64)v17, v5);
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
        v5 = v26;
        result = v28;
        v22 = v30;
        v17 = v34;
        v14 = v39;
        v25 = (void *)(*(_QWORD *)a1 + 16LL * (unsigned int)v11);
      }
      if ( v23 != v17 )
      {
        v27 = v5;
        v29 = v22;
        v33 = v17;
        v37 = v14;
        result = (char *)memmove(v25, v23, (size_t)result);
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
        v5 = v27;
        v22 = v29;
        v17 = v33;
        v14 = v37;
      }
      *(_DWORD *)(a1 + 8) = v43 + v11;
      if ( v23 != v8 )
      {
        v44 = v5;
        result = (char *)memmove(&v17[-(v22 - v14)], v8, v22 - v14);
        v5 = v44;
      }
      if ( v9 != a4 )
        return (char *)memmove(v8, v9, v5);
    }
    else
    {
      v21 = v6 + v11;
      *(_DWORD *)(a1 + 8) = v21;
      if ( v17 != v8 )
      {
        v31 = (result - v14) >> 4;
        v35 = v17;
        v40 = (char *)(result - v14);
        result = (char *)memcpy((void *)(16LL * v21 - v18 + v12), v8, v18);
        v19 = v31;
        v17 = v35;
        v18 = (size_t)v40;
      }
      if ( v19 )
      {
        result = 0;
        do
        {
          *(__m128i *)&result[(_QWORD)v8] = _mm_loadu_si128((const __m128i *)&result[(_QWORD)v9]);
          result += 16;
          --v20;
        }
        while ( v20 );
        v9 += v18;
      }
      if ( a4 != v9 )
        return (char *)memcpy(v17, v9, a4 - v9);
    }
  }
  return result;
}
