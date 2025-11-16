// Function: sub_2957A00
// Address: 0x2957a00
//
__int64 __fastcall sub_2957A00(__int64 a1, char *a2, size_t a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r11
  __int64 v7; // r10
  __int64 v8; // r9
  size_t v9; // r15
  char *v11; // r12
  __int64 v13; // rdi
  __int64 v14; // rsi
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  __int64 result; // rax
  __int64 v18; // r13
  __int64 v19; // r8
  size_t v20; // rdx
  __int64 v21; // rcx
  unsigned int v22; // esi
  __int64 v23; // r8
  size_t v24; // rax
  char *v25; // r15
  void *v26; // rdi
  __int64 v27; // r11
  unsigned __int64 v28; // rdx
  __int64 v29; // [rsp+0h] [rbp-60h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  int v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  size_t v37; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+18h] [rbp-48h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+20h] [rbp-40h]
  __int64 v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+20h] [rbp-40h]
  __int64 v48; // [rsp+20h] [rbp-40h]
  char *v49; // [rsp+28h] [rbp-38h]
  size_t v50; // [rsp+28h] [rbp-38h]
  __int64 v51; // [rsp+28h] [rbp-38h]

  v6 = a4;
  v7 = a4 - a3;
  v8 = (__int64)(a4 - a3) >> 3;
  v9 = a3;
  v11 = a2;
  v13 = *(_QWORD *)a1;
  v14 = *(unsigned int *)(a1 + 8);
  v15 = *(unsigned int *)(a1 + 12);
  v16 = v14 + v8;
  v49 = &a2[-v13];
  result = 8 * v14;
  v18 = v13 + 8 * v14;
  if ( v11 == (char *)v18 )
  {
    if ( v16 > v15 )
    {
      v47 = v8;
      v51 = v7;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 8u, a5, v8);
      v14 = *(unsigned int *)(a1 + 8);
      result = *(_QWORD *)a1;
      v8 = v47;
      v7 = v51;
      v18 = *(_QWORD *)a1 + 8 * v14;
    }
    if ( v7 > 0 )
    {
      result = 0;
      do
      {
        *(_QWORD *)(v18 + 8 * result) = *(_QWORD *)(a3 + 8 * result);
        ++result;
      }
      while ( v8 - result > 0 );
      LODWORD(v14) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v8 + v14;
  }
  else
  {
    v19 = v8;
    if ( v16 > v15 )
    {
      v31 = v8;
      v35 = v6;
      v39 = v7;
      v44 = v8;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 8u, v8, v8);
      v13 = *(_QWORD *)a1;
      v14 = *(unsigned int *)(a1 + 8);
      v8 = v31;
      v11 = &v49[*(_QWORD *)a1];
      v6 = v35;
      v7 = v39;
      result = 8 * v14;
      v19 = v44;
      v18 = *(_QWORD *)a1 + 8 * v14;
    }
    v20 = result - (_QWORD)v49;
    v21 = (result - (__int64)v49) >> 3;
    if ( result - (__int64)v49 >= (unsigned __int64)v7 )
    {
      v23 = result - v7;
      v24 = v7;
      v25 = (char *)(v13 + v23);
      v26 = (void *)v18;
      v27 = v7 >> 3;
      v28 = (v7 >> 3) + v14;
      if ( v28 > *(unsigned int *)(a1 + 12) )
      {
        v29 = v7 >> 3;
        v33 = v8;
        v37 = v7;
        v42 = v7;
        v48 = v23;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v28, 8u, v23, v8);
        v14 = *(unsigned int *)(a1 + 8);
        LODWORD(v27) = v29;
        v8 = v33;
        v24 = v37;
        v7 = v42;
        v26 = (void *)(*(_QWORD *)a1 + 8 * v14);
        v23 = v48;
      }
      if ( v25 != (char *)v18 )
      {
        v32 = v27;
        v36 = v8;
        v40 = v7;
        v45 = v23;
        memmove(v26, v25, v24);
        LODWORD(v14) = *(_DWORD *)(a1 + 8);
        LODWORD(v27) = v32;
        v8 = v36;
        v7 = v40;
        v23 = v45;
      }
      *(_DWORD *)(a1 + 8) = v27 + v14;
      if ( v25 != v11 )
      {
        v41 = v8;
        v46 = v7;
        memmove((void *)(v18 - (v23 - (_QWORD)v49)), v11, v23 - (_QWORD)v49);
        v8 = v41;
        v7 = v46;
      }
      result = 0;
      if ( v7 > 0 )
      {
        do
        {
          *(_QWORD *)&v11[8 * result] = *(_QWORD *)(a3 + 8 * result);
          ++result;
        }
        while ( v8 - result > 0 );
      }
    }
    else
    {
      v22 = v8 + v14;
      *(_DWORD *)(a1 + 8) = v22;
      if ( v11 != (char *)v18 )
      {
        v30 = (result - (__int64)v49) >> 3;
        v34 = v6;
        v38 = v7;
        v43 = v19;
        v50 = result - (_QWORD)v49;
        result = (__int64)memcpy((void *)(v13 + 8LL * v22 - v20), v11, v20);
        v21 = v30;
        v6 = v34;
        v7 = v38;
        v19 = v43;
        v20 = v50;
      }
      if ( v21 )
      {
        for ( result = 0; result != v21; ++result )
          *(_QWORD *)&v11[8 * result] = *(_QWORD *)(a3 + 8 * result);
        v9 = a3 + v20;
        v7 = v6 - (a3 + v20);
        v19 = v7 >> 3;
      }
      if ( v7 > 0 )
      {
        result = 0;
        do
        {
          *(_QWORD *)(v18 + 8 * result) = *(_QWORD *)(v9 + 8 * result);
          ++result;
        }
        while ( v19 - result > 0 );
      }
    }
  }
  return result;
}
