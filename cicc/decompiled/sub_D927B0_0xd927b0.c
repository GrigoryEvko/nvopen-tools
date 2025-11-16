// Function: sub_D927B0
// Address: 0xd927b0
//
__int64 __fastcall sub_D927B0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r9
  __int64 v6; // r8
  __int64 v8; // r15
  char *v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  __int64 result; // rax
  char *v15; // r10
  unsigned __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // r11
  size_t v19; // rdx
  __int64 v20; // rcx
  unsigned int v21; // esi
  __int64 v22; // r11
  size_t v23; // rax
  char *v24; // r9
  unsigned __int64 v25; // rdx
  void *v26; // rdi
  size_t v27; // [rsp+0h] [rbp-60h]
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  char *v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  char *v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  char *v40; // [rsp+20h] [rbp-40h]
  char *v41; // [rsp+20h] [rbp-40h]
  size_t v42; // [rsp+28h] [rbp-38h]
  char *v43; // [rsp+28h] [rbp-38h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  __int64 v45; // [rsp+28h] [rbp-38h]
  __int64 v46; // [rsp+28h] [rbp-38h]

  v5 = a4;
  v6 = a4 - a3;
  v8 = (a4 - a3) >> 3;
  v9 = a2;
  v10 = a3;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v13 = *(unsigned int *)(a1 + 12);
  result = 8 * v11;
  v15 = &a2[-v12];
  v16 = v11 + v8;
  v17 = v12 + 8 * v11;
  if ( v9 == (char *)v17 )
  {
    if ( v16 > v13 )
    {
      v46 = v6;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 8u, v6, v5);
      v11 = *(unsigned int *)(a1 + 8);
      result = *(_QWORD *)a1;
      v6 = v46;
      v17 = *(_QWORD *)a1 + 8 * v11;
    }
    if ( v6 > 0 )
    {
      result = 0;
      do
      {
        *(_QWORD *)(v17 + 8 * result) = *(_QWORD *)(v10 + 8 * result);
        ++result;
      }
      while ( v8 - result > 0 );
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v8 + v11;
  }
  else
  {
    v18 = v8;
    if ( v16 > v13 )
    {
      v32 = v5;
      v39 = v6;
      v43 = v15;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 8u, v6, v5);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_QWORD *)a1;
      v15 = v43;
      v5 = v32;
      result = 8 * v11;
      v18 = v8;
      v6 = v39;
      v9 = &v43[*(_QWORD *)a1];
      v17 = *(_QWORD *)a1 + 8 * v11;
    }
    v19 = result - (_QWORD)v15;
    v20 = (result - (__int64)v15) >> 3;
    if ( result - (__int64)v15 >= (unsigned __int64)v6 )
    {
      v22 = result - v6;
      v23 = v6;
      v24 = (char *)(v12 + v22);
      v25 = (v6 >> 3) + v11;
      v44 = v6 >> 3;
      v26 = (void *)v17;
      if ( v25 > *(unsigned int *)(a1 + 12) )
      {
        v27 = v6;
        v30 = v6;
        v34 = v24;
        v37 = v22;
        v41 = v15;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v25, 8u, v6, (__int64)v24);
        v11 = *(unsigned int *)(a1 + 8);
        v23 = v27;
        v6 = v30;
        v24 = v34;
        v22 = v37;
        v26 = (void *)(*(_QWORD *)a1 + 8 * v11);
        v15 = v41;
      }
      if ( v24 != (char *)v17 )
      {
        v29 = v6;
        v33 = v22;
        v36 = v15;
        v40 = v24;
        memmove(v26, v24, v23);
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
        v6 = v29;
        v22 = v33;
        v15 = v36;
        v24 = v40;
      }
      *(_DWORD *)(a1 + 8) = v44 + v11;
      if ( v24 != v9 )
      {
        v45 = v6;
        memmove((void *)(v17 - (v22 - (_QWORD)v15)), v9, v22 - (_QWORD)v15);
        v6 = v45;
      }
      result = 0;
      if ( v6 > 0 )
      {
        do
        {
          *(_QWORD *)&v9[8 * result] = *(_QWORD *)(v10 + 8 * result);
          ++result;
        }
        while ( v8 - result > 0 );
      }
    }
    else
    {
      v21 = v8 + v11;
      *(_DWORD *)(a1 + 8) = v21;
      if ( (char *)v17 != v9 )
      {
        v28 = (result - (__int64)v15) >> 3;
        v31 = v5;
        v35 = v18;
        v38 = v6;
        v42 = result - (_QWORD)v15;
        result = (__int64)memcpy((void *)(v12 + 8LL * v21 - v19), v9, v19);
        v20 = v28;
        v5 = v31;
        v18 = v35;
        v6 = v38;
        v19 = v42;
      }
      if ( v20 )
      {
        for ( result = 0; result != v20; ++result )
          *(_QWORD *)&v9[8 * result] = *(_QWORD *)(v10 + 8 * result);
        v10 += v19;
        v6 = v5 - v10;
        v18 = (v5 - v10) >> 3;
      }
      if ( v6 > 0 )
      {
        result = 0;
        do
        {
          *(_QWORD *)(v17 + 8 * result) = *(_QWORD *)(v10 + 8 * result);
          ++result;
        }
        while ( v18 - result > 0 );
      }
    }
  }
  return result;
}
