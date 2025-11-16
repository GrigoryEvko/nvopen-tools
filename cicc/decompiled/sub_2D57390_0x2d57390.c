// Function: sub_2D57390
// Address: 0x2d57390
//
__int64 __fastcall sub_2D57390(__int64 a1, char *a2, char *a3, _QWORD *a4)
{
  _QWORD *v5; // r11
  __int64 v6; // r10
  unsigned __int64 v7; // rax
  char *v9; // r13
  _QWORD *v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // r8
  unsigned __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // r9
  unsigned __int64 v16; // rdx
  char *v17; // r12
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // rcx
  unsigned int v20; // eax
  unsigned __int64 v21; // rdx
  __int64 result; // rax
  __int64 v23; // r11
  size_t v24; // rax
  char *v25; // r8
  __int64 v26; // rdi
  unsigned __int64 v27; // rdx
  char *v28; // rdi
  __int64 v29; // [rsp+0h] [rbp-60h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  size_t v31; // [rsp+8h] [rbp-58h]
  unsigned __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  char *v34; // [rsp+10h] [rbp-50h]
  _QWORD *v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  char *v40; // [rsp+20h] [rbp-40h]
  unsigned __int64 v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  _QWORD *v43; // [rsp+28h] [rbp-38h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  int v45; // [rsp+28h] [rbp-38h]
  __int64 v46; // [rsp+28h] [rbp-38h]
  _QWORD *v47; // [rsp+28h] [rbp-38h]

  v5 = a4;
  v6 = (char *)a4 - a3;
  v7 = ((char *)a4 - a3) >> 5;
  v9 = a2;
  v10 = a3;
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v13 = *(unsigned int *)(a1 + 12);
  v14 = 8 * v11;
  v15 = (__int64)&a2[-v12];
  v16 = v11 + v7;
  v17 = (char *)(v12 + 8 * v11);
  if ( v9 == v17 )
  {
    if ( v16 > v13 )
    {
      v41 = v7;
      v47 = v5;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 8u, v12, v15);
      v11 = *(unsigned int *)(a1 + 8);
      v7 = v41;
      v5 = v47;
      v17 = (char *)(*(_QWORD *)a1 + 8 * v11);
    }
    if ( v10 != v5 )
    {
      do
      {
        if ( v17 )
          *(_QWORD *)v17 = *v10;
        v10 += 4;
        v17 += 8;
      }
      while ( v5 != v10 );
      v11 = *(unsigned int *)(a1 + 8);
    }
    result = v11 + v7;
    *(_DWORD *)(a1 + 8) = result;
  }
  else
  {
    v18 = v7;
    if ( v16 > v13 )
    {
      v32 = v7;
      v35 = v5;
      v39 = v6;
      v44 = v15;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 8u, v12, v15);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_QWORD *)a1;
      v15 = v44;
      v7 = v32;
      v14 = 8 * v11;
      v5 = v35;
      v6 = v39;
      v9 = (char *)(*(_QWORD *)a1 + v44);
      v17 = (char *)(*(_QWORD *)a1 + 8 * v11);
    }
    v19 = (v14 - v15) >> 3;
    if ( v19 >= v7 )
    {
      v23 = 8 * (v11 - v7);
      v24 = v14 - v23;
      v25 = (char *)(v23 + v12);
      v26 = (v14 - v23) >> 3;
      v27 = v11 + v26;
      v45 = v26;
      v28 = v17;
      if ( v27 > *(unsigned int *)(a1 + 12) )
      {
        v29 = v6;
        v31 = v24;
        v34 = v25;
        v37 = v23;
        v42 = v15;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v27, 8u, (__int64)v25, v15);
        v11 = *(unsigned int *)(a1 + 8);
        v6 = v29;
        v24 = v31;
        v25 = v34;
        v23 = v37;
        v28 = (char *)(*(_QWORD *)a1 + 8 * v11);
        v15 = v42;
      }
      if ( v25 != v17 )
      {
        v30 = v6;
        v33 = v23;
        v36 = v15;
        v40 = v25;
        memmove(v28, v25, v24);
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
        v6 = v30;
        v23 = v33;
        v15 = v36;
        v25 = v40;
      }
      *(_DWORD *)(a1 + 8) = v45 + v11;
      if ( v25 != v9 )
      {
        v46 = v6;
        memmove(&v17[-(v23 - v15)], v9, v23 - v15);
        v6 = v46;
      }
      result = 0;
      if ( v6 > 0 )
      {
        do
        {
          *(_QWORD *)&v9[result] = *(_QWORD *)((char *)v10 + 4 * result);
          result += 8;
          --v18;
        }
        while ( v18 );
      }
    }
    else
    {
      v20 = v11 + v7;
      *(_DWORD *)(a1 + 8) = v20;
      if ( v9 != v17 )
      {
        v38 = (v14 - v15) >> 3;
        v43 = v5;
        memcpy((void *)(v12 + 8LL * v20 - (v14 - v15)), v9, v14 - v15);
        v19 = v38;
        v5 = v43;
      }
      v21 = v19;
      result = 0;
      if ( !v19 )
        goto LABEL_13;
      do
      {
        *(_QWORD *)&v9[result] = *(_QWORD *)((char *)v10 + 4 * result);
        result += 8;
        --v21;
      }
      while ( v21 );
      v10 += 4 * v19;
      while ( v5 != v10 )
      {
        if ( v17 )
        {
          result = *v10;
          *(_QWORD *)v17 = *v10;
        }
        v10 += 4;
        v17 += 8;
LABEL_13:
        ;
      }
    }
  }
  return result;
}
