// Function: sub_1526BE0
// Address: 0x1526be0
//
char *__fastcall sub_1526BE0(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rsi
  char *v10; // rdi
  char *v11; // r13
  __int64 v12; // r8
  char *v13; // rdx
  char *result; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  char *v17; // r15
  char *v18; // rsi
  __int64 v19; // rcx
  char *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // r14
  int v23; // r15d
  __int64 v24; // rax
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // rax
  bool v27; // cf
  unsigned __int64 v28; // rcx
  _QWORD *v29; // r14
  char *v30; // rcx
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rdi
  char *v35; // rcx
  _QWORD *v36; // rax
  __int64 v37; // rsi
  char *v38; // r12
  _QWORD *v39; // r15
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // r13
  volatile signed __int32 *v43; // rdi
  __int64 v44; // r12
  __int64 v45; // rax
  __int64 v46; // [rsp+8h] [rbp-48h]
  int v47; // [rsp+14h] [rbp-3Ch] BYREF
  unsigned __int64 v48[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_1524D80(a1, 1u, *((_DWORD *)a1 + 4));
  sub_1524E40(a1, a2, 8);
  sub_1524E40(a1, a3, 4);
  if ( *((_DWORD *)a1 + 2) )
  {
    v22 = *a1;
    v23 = *((_DWORD *)a1 + 3);
    v24 = *(unsigned int *)(*a1 + 8LL);
    if ( (unsigned __int64)*(unsigned int *)(*a1 + 12LL) - v24 <= 3 )
    {
      sub_16CD150(*a1, v22 + 16, v24 + 4, 1);
      v24 = *(unsigned int *)(v22 + 8);
    }
    *(_DWORD *)(*(_QWORD *)v22 + v24) = v23;
    *(_DWORD *)(v22 + 8) += 4;
    a1[1] = 0;
  }
  v48[0] = (unsigned __int64)*(unsigned int *)(*a1 + 8LL) >> 2;
  v47 = *((_DWORD *)a1 + 4);
  sub_1524D80(a1, 0, 32);
  *((_DWORD *)a1 + 4) = a3;
  v6 = a1[7];
  if ( v6 == a1[8] )
  {
    sub_1525D70(a1 + 6, (char *)v6, &v47, v48);
    v9 = a1[7];
  }
  else
  {
    if ( v6 )
    {
      v7 = v48[0];
      v8 = v47;
      *(_QWORD *)(v6 + 16) = 0;
      *(_QWORD *)(v6 + 24) = 0;
      *(_DWORD *)v6 = v8;
      *(_QWORD *)(v6 + 8) = v7;
      *(_QWORD *)(v6 + 32) = 0;
      v6 = a1[7];
    }
    v9 = v6 + 40;
    a1[7] = v9;
  }
  v10 = *(char **)(v9 - 24);
  v11 = *(char **)(v9 - 16);
  v12 = *(_QWORD *)(v9 - 8);
  *(_QWORD *)(v9 - 24) = a1[3];
  *(_QWORD *)(v9 - 16) = a1[4];
  *(_QWORD *)(v9 - 8) = a1[5];
  v13 = (char *)a1[10];
  result = (char *)a1[9];
  a1[3] = v10;
  a1[4] = v11;
  a1[5] = v12;
  if ( v13 != result && a2 == *((_DWORD *)v13 - 8) )
  {
    v17 = (char *)*((_QWORD *)v13 - 2);
    v18 = (char *)*((_QWORD *)v13 - 3);
    if ( v18 == v17 )
      return result;
LABEL_13:
    v19 = v17 - v18;
    if ( v12 - (__int64)v11 >= (unsigned __int64)(v17 - v18) )
    {
      result = v18;
      v20 = &v11[v19];
      do
      {
        if ( v11 )
        {
          *(_QWORD *)v11 = *(_QWORD *)result;
          v21 = *((_QWORD *)result + 1);
          *((_QWORD *)v11 + 1) = v21;
          if ( v21 )
          {
            if ( &_pthread_key_create )
              _InterlockedAdd((volatile signed __int32 *)(v21 + 8), 1u);
            else
              ++*(_DWORD *)(v21 + 8);
          }
        }
        v11 += 16;
        result += 16;
      }
      while ( v20 != v11 );
      a1[4] += v19;
      return result;
    }
    v25 = v19 >> 4;
    v26 = (v11 - v10) >> 4;
    if ( v25 > 0x7FFFFFFFFFFFFFFLL - v26 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v25 < v26 )
      v25 = (v11 - v10) >> 4;
    v27 = __CFADD__(v26, v25);
    v28 = v26 + v25;
    if ( v27 )
    {
      v44 = 0x7FFFFFFFFFFFFFF0LL;
    }
    else
    {
      if ( !v28 )
      {
        v46 = 0;
        v29 = 0;
LABEL_34:
        if ( v11 == v10 )
        {
          v32 = v29;
        }
        else
        {
          v30 = v10;
          v31 = v29;
          v32 = (_QWORD *)((char *)v29 + v11 - v10);
          do
          {
            if ( v31 )
            {
              v33 = *(_QWORD *)v30;
              v31[1] = 0;
              *v31 = v33;
              v34 = *((_QWORD *)v30 + 1);
              *((_QWORD *)v30 + 1) = 0;
              v31[1] = v34;
              *(_QWORD *)v30 = 0;
            }
            v31 += 2;
            v30 += 16;
          }
          while ( v32 != v31 );
        }
        v35 = v18;
        v36 = (_QWORD *)((char *)v32 + v17 - v18);
        do
        {
          if ( v32 )
          {
            *v32 = *(_QWORD *)v35;
            v37 = *((_QWORD *)v35 + 1);
            v32[1] = v37;
            if ( v37 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd((volatile signed __int32 *)(v37 + 8), 1u);
              else
                ++*(_DWORD *)(v37 + 8);
            }
          }
          v32 += 2;
          v35 += 16;
        }
        while ( v36 != v32 );
        v38 = (char *)a1[4];
        if ( v11 == v38 )
        {
          v39 = v36;
        }
        else
        {
          v39 = (_QWORD *)((char *)v36 + v38 - v11);
          do
          {
            if ( v36 )
            {
              v40 = *(_QWORD *)v11;
              v36[1] = 0;
              *v36 = v40;
              v41 = *((_QWORD *)v11 + 1);
              *((_QWORD *)v11 + 1) = 0;
              v36[1] = v41;
              *(_QWORD *)v11 = 0;
            }
            v36 += 2;
            v11 += 16;
          }
          while ( v36 != v39 );
          v38 = (char *)a1[4];
        }
        v42 = a1[3];
        if ( (char *)v42 != v38 )
        {
          do
          {
            v43 = *(volatile signed __int32 **)(v42 + 8);
            if ( v43 )
              sub_A191D0(v43);
            v42 += 16;
          }
          while ( v38 != (char *)v42 );
          v38 = (char *)a1[3];
        }
        if ( v38 )
          j_j___libc_free_0(v38, a1[5] - (_QWORD)v38);
        a1[3] = v29;
        a1[4] = v39;
        a1[5] = v46;
        return (char *)v46;
      }
      if ( v28 > 0x7FFFFFFFFFFFFFFLL )
        v28 = 0x7FFFFFFFFFFFFFFLL;
      v44 = 16 * v28;
    }
    v45 = sub_22077B0(v44);
    v10 = (char *)a1[3];
    v29 = (_QWORD *)v45;
    v46 = v44 + v45;
    goto LABEL_34;
  }
  v15 = (v13 - result) >> 5;
  if ( (_DWORD)v15 )
  {
    v16 = (__int64)&result[32 * (unsigned int)(v15 - 1) + 32];
    while ( a2 != *(_DWORD *)result )
    {
      result += 32;
      if ( (char *)v16 == result )
        return result;
    }
    v17 = (char *)*((_QWORD *)result + 2);
    v18 = (char *)*((_QWORD *)result + 1);
    if ( v18 != v17 )
      goto LABEL_13;
  }
  return result;
}
