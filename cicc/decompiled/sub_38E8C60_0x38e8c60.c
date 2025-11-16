// Function: sub_38E8C60
// Address: 0x38e8c60
//
char *__fastcall sub_38E8C60(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v8; // r15
  char *v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  char *v15; // r10
  __int64 v16; // r13
  __int64 v17; // r11
  unsigned __int64 v18; // rcx
  unsigned int v19; // esi
  __int64 i; // rax
  __int64 v21; // rax
  __int64 v23; // rax
  __int64 v24; // r11
  signed __int64 v25; // rax
  char *v26; // r8
  void *v27; // rdi
  __int64 v28; // rax
  signed __int64 v29; // [rsp+0h] [rbp-60h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  char *v37; // [rsp+18h] [rbp-48h]
  char *v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  char *v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  char *v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  char *v45; // [rsp+28h] [rbp-38h]
  __int64 v46; // [rsp+28h] [rbp-38h]
  __int64 v47; // [rsp+28h] [rbp-38h]
  char *v48; // [rsp+28h] [rbp-38h]

  v5 = a4;
  v6 = a4 - a3;
  v8 = (a4 - a3) >> 1;
  v9 = a2;
  v10 = a3;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v13 = *(unsigned int *)(a1 + 12);
  v14 = 4 * v11;
  v15 = &a2[-v12];
  v16 = v12 + 4 * v11;
  if ( v9 == (char *)v16 )
  {
    if ( v13 - v11 < v8 )
    {
      v42 = v6;
      v48 = v15;
      sub_16CD150(a1, (const void *)(a1 + 16), v11 + v8, 4, v5, v6);
      v12 = *(_QWORD *)a1;
      v11 = *(unsigned int *)(a1 + 8);
      v6 = v42;
      v15 = v48;
      v16 = *(_QWORD *)a1 + 4 * v11;
    }
    if ( v6 > 0 )
    {
      v23 = 0;
      do
      {
        *(_DWORD *)(v16 + 4 * v23) = *(unsigned __int16 *)(v10 + 2 * v23);
        ++v23;
      }
      while ( (__int64)(v8 - v23) > 0 );
      v12 = *(_QWORD *)a1;
      LODWORD(v11) = *(_DWORD *)(a1 + 8);
    }
    v9 = &v15[v12];
    *(_DWORD *)(a1 + 8) = v8 + v11;
  }
  else
  {
    v17 = v8;
    if ( v11 + v8 > v13 )
    {
      v33 = v5;
      v40 = v6;
      v45 = v15;
      sub_16CD150(a1, (const void *)(a1 + 16), v11 + v8, 4, v5, v6);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_QWORD *)a1;
      v15 = v45;
      v5 = v33;
      v14 = 4 * v11;
      v17 = v8;
      v6 = v40;
      v9 = &v45[*(_QWORD *)a1];
      v16 = *(_QWORD *)a1 + 4 * v11;
    }
    v18 = (v14 - (__int64)v15) >> 2;
    if ( v18 >= v8 )
    {
      v24 = 4 * (v11 - v8);
      v25 = v14 - v24;
      v26 = (char *)(v12 + v24);
      v27 = (void *)v16;
      v46 = v25 >> 2;
      if ( v25 >> 2 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v11 )
      {
        v29 = v25;
        v31 = v6;
        v35 = 4 * (v11 - v8);
        v38 = v26;
        v43 = v15;
        sub_16CD150(a1, (const void *)(a1 + 16), v11 + (v25 >> 2), 4, (int)v26, v6);
        v11 = *(unsigned int *)(a1 + 8);
        v25 = v29;
        v6 = v31;
        v24 = v35;
        v26 = v38;
        v27 = (void *)(*(_QWORD *)a1 + 4 * v11);
        v15 = v43;
      }
      if ( v26 != (char *)v16 )
      {
        v30 = v6;
        v34 = v24;
        v37 = v15;
        v41 = v26;
        memmove(v27, v26, v25);
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
        v6 = v30;
        v24 = v34;
        v15 = v37;
        v26 = v41;
      }
      *(_DWORD *)(a1 + 8) = v46 + v11;
      if ( v26 != v9 )
      {
        v47 = v6;
        memmove((void *)(v16 - (v24 - (_QWORD)v15)), v9, v24 - (_QWORD)v15);
        v6 = v47;
      }
      if ( v6 > 0 )
      {
        v28 = 0;
        do
        {
          *(_DWORD *)&v9[4 * v28] = *(unsigned __int16 *)(v10 + 2 * v28);
          ++v28;
        }
        while ( (__int64)(v8 - v28) > 0 );
      }
    }
    else
    {
      v19 = v8 + v11;
      *(_DWORD *)(a1 + 8) = v19;
      if ( v9 != (char *)v16 )
      {
        v32 = (v14 - (__int64)v15) >> 2;
        v36 = v5;
        v39 = v17;
        v44 = v6;
        memcpy((void *)(v12 + 4LL * v19 - (v14 - (_QWORD)v15)), v9, v14 - (_QWORD)v15);
        v18 = v32;
        v5 = v36;
        v17 = v39;
        v6 = v44;
      }
      if ( v18 )
      {
        for ( i = 0; i != v18; ++i )
          *(_DWORD *)&v9[4 * i] = *(unsigned __int16 *)(v10 + 2 * i);
        v10 += 2 * v18;
        v6 = v5 - v10;
        v17 = (v5 - v10) >> 1;
      }
      if ( v6 > 0 )
      {
        v21 = 0;
        do
        {
          *(_DWORD *)(v16 + 4 * v21) = *(unsigned __int16 *)(v10 + 2 * v21);
          ++v21;
        }
        while ( v17 - v21 > 0 );
      }
    }
  }
  return v9;
}
