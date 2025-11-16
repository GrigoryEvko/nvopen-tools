// Function: sub_145C820
// Address: 0x145c820
//
char *__fastcall sub_145C820(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r10
  __int64 v7; // r14
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rcx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r11
  size_t v16; // r11
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // r13
  unsigned int v19; // eax
  char *v20; // r15
  __int64 v22; // rax
  unsigned __int64 v23; // r13
  __int64 v24; // r11
  char *v25; // r13
  void *v26; // rdi
  signed __int64 v27; // r9
  unsigned __int64 v28; // r8
  char *v29; // rbx
  __int64 v30; // [rsp+0h] [rbp-60h]
  int v31; // [rsp+8h] [rbp-58h]
  signed __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  unsigned __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  char *v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  char *src; // [rsp+28h] [rbp-38h]

  v5 = a3;
  v7 = a3;
  v9 = *a1;
  src = a2;
  v40 = &a2[-v9];
  v10 = *((unsigned int *)a1 + 2);
  v11 = 8 * v10;
  v12 = v9 + 8 * v10;
  if ( a2 == (char *)v12 )
  {
    if ( v5 == a4 )
    {
      LODWORD(v23) = 0;
    }
    else
    {
      v22 = v5;
      v23 = 0;
      do
      {
        v22 = *(_QWORD *)(v22 + 8);
        ++v23;
      }
      while ( v22 != a4 );
      if ( (unsigned __int64)*((unsigned int *)a1 + 3) - v10 < v23 )
      {
        sub_16CD150(a1, a1 + 2, v10 + v23, 8);
        v12 = *a1 + 8LL * *((unsigned int *)a1 + 2);
      }
      do
      {
        v12 += 8;
        *(_QWORD *)(v12 - 8) = sub_1648700(v7);
        v7 = *(_QWORD *)(v7 + 8);
      }
      while ( v7 != a4 );
      v9 = *a1;
      LODWORD(v10) = *((_DWORD *)a1 + 2);
    }
    *((_DWORD *)a1 + 2) = v23 + v10;
    return &v40[v9];
  }
  else
  {
    if ( v5 == a4 )
    {
      v15 = *((unsigned int *)a1 + 2);
      v14 = 0;
    }
    else
    {
      v13 = v5;
      v14 = 0;
      do
      {
        v13 = *(_QWORD *)(v13 + 8);
        ++v14;
      }
      while ( v13 != a4 );
      v15 = v10 + v14;
    }
    if ( v15 > *((unsigned int *)a1 + 3) )
    {
      v33 = v5;
      v36 = v14;
      sub_16CD150(a1, a1 + 2, v15, 8);
      v9 = *a1;
      v10 = *((unsigned int *)a1 + 2);
      v5 = v33;
      v11 = 8 * v10;
      src = &v40[*a1];
      v14 = v36;
      v12 = *a1 + 8 * v10;
    }
    v16 = v11 - (_QWORD)v40;
    v17 = (v11 - (__int64)v40) >> 3;
    v18 = v17;
    if ( v17 >= v14 )
    {
      v24 = 8 * (v10 - v14);
      v25 = (char *)(v9 + v24);
      v26 = (void *)v12;
      v27 = v11 - v24;
      v28 = (v11 - v24) >> 3;
      if ( v28 > (unsigned __int64)*((unsigned int *)a1 + 3) - v10 )
      {
        v30 = v5;
        v32 = v11 - v24;
        v35 = 8 * (v10 - v14);
        v39 = v27 >> 3;
        sub_16CD150(a1, a1 + 2, v10 + v28, 8);
        v10 = *((unsigned int *)a1 + 2);
        v5 = v30;
        v27 = v32;
        v24 = v35;
        LODWORD(v28) = v39;
        v26 = (void *)(*a1 + 8 * v10);
      }
      if ( v25 != (char *)v12 )
      {
        v31 = v28;
        v34 = v5;
        v37 = v24;
        memmove(v26, v25, v27);
        LODWORD(v10) = *((_DWORD *)a1 + 2);
        LODWORD(v28) = v31;
        v5 = v34;
        v24 = v37;
      }
      *((_DWORD *)a1 + 2) = v28 + v10;
      if ( v25 != src )
      {
        v38 = v5;
        memmove((void *)(v12 - (v24 - (_QWORD)v40)), src, v24 - (_QWORD)v40);
        v5 = v38;
      }
      v29 = src;
      if ( v5 != a4 )
      {
        do
        {
          v29 += 8;
          *((_QWORD *)v29 - 1) = sub_1648700(v7);
          v7 = *(_QWORD *)(v7 + 8);
        }
        while ( v7 != a4 );
      }
    }
    else
    {
      v19 = v10 + v14;
      *((_DWORD *)a1 + 2) = v19;
      if ( src != (char *)v12 )
      {
        v41 = (v11 - (__int64)v40) >> 3;
        memcpy((void *)(v9 + 8LL * v19 - v16), src, v16);
        v17 = v41;
      }
      if ( !v17 )
        goto LABEL_16;
      v20 = src;
      do
      {
        v20 += 8;
        *((_QWORD *)v20 - 1) = sub_1648700(v7);
        v7 = *(_QWORD *)(v7 + 8);
        --v18;
      }
      while ( v18 );
      while ( v7 != a4 )
      {
        v12 += 8;
        *(_QWORD *)(v12 - 8) = sub_1648700(v7);
        v7 = *(_QWORD *)(v7 + 8);
LABEL_16:
        ;
      }
    }
    return src;
  }
}
