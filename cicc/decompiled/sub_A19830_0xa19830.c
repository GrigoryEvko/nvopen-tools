// Function: sub_A19830
// Address: 0xa19830
//
char *__fastcall sub_A19830(__int64 a1, unsigned int a2, unsigned int a3)
{
  _QWORD *v6; // r12
  unsigned __int64 v7; // r15
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rsi
  char *v12; // rdi
  char *v13; // r12
  __int64 v14; // r8
  char *v15; // rdx
  char *result; // rax
  char *v17; // rcx
  char *v18; // rsi
  char *v19; // r15
  __int64 v20; // rcx
  char *v21; // rsi
  __int64 v22; // rdx
  _QWORD *v23; // r12
  int v24; // r15d
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // r13
  unsigned __int64 v28; // rax
  bool v29; // cf
  unsigned __int64 v30; // rcx
  __int64 v31; // r13
  __int64 v32; // rax
  _QWORD *v33; // r14
  char *v34; // rcx
  _QWORD *v35; // rax
  _QWORD *v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rdi
  char *v39; // rcx
  _QWORD *v40; // rax
  __int64 v41; // rsi
  char *v42; // r13
  _QWORD *v43; // r15
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // r12
  volatile signed __int32 *v47; // rdi
  __int64 v48; // [rsp+8h] [rbp-48h]
  int v49; // [rsp+14h] [rbp-3Ch] BYREF
  unsigned __int64 v50[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_A17B10(a1, 1u, *(_DWORD *)(a1 + 56));
  sub_A17CC0(a1, a2, 8);
  sub_A17CC0(a1, a3, 4);
  if ( *(_DWORD *)(a1 + 48) )
  {
    v23 = *(_QWORD **)(a1 + 24);
    v24 = *(_DWORD *)(a1 + 52);
    v25 = v23[1];
    if ( (unsigned __int64)(v25 + 4) > v23[2] )
    {
      sub_C8D290(*(_QWORD *)(a1 + 24), v23 + 3, v25 + 4, 1);
      v25 = v23[1];
    }
    *(_DWORD *)(*v23 + v25) = v24;
    v23[1] += 4LL;
    *(_QWORD *)(a1 + 48) = 0;
  }
  v6 = *(_QWORD **)(a1 + 32);
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
  if ( v6 && (unsigned __int8)sub_CB7440(*(_QWORD *)(a1 + 32)) )
  {
    if ( !(unsigned __int8)sub_CB7440(v6) )
      BUG();
    v7 += v6[4] - v6[2] + (*(__int64 (__fastcall **)(_QWORD *))(*v6 + 80LL))(v6);
  }
  v50[0] = v7 >> 2;
  v49 = *(_DWORD *)(a1 + 56);
  sub_A17B10(a1, 0, 32);
  *(_DWORD *)(a1 + 56) = a3;
  v8 = *(_QWORD *)(a1 + 112);
  if ( v8 == *(_QWORD *)(a1 + 120) )
  {
    sub_A18BD0((__int64 *)(a1 + 104), (char *)v8, &v49, v50);
    v11 = *(_QWORD *)(a1 + 112);
  }
  else
  {
    if ( v8 )
    {
      v9 = v50[0];
      v10 = v49;
      *(_QWORD *)(v8 + 16) = 0;
      *(_QWORD *)(v8 + 24) = 0;
      *(_DWORD *)v8 = v10;
      *(_QWORD *)(v8 + 8) = v9;
      *(_QWORD *)(v8 + 32) = 0;
      v8 = *(_QWORD *)(a1 + 112);
    }
    v11 = v8 + 40;
    *(_QWORD *)(a1 + 112) = v11;
  }
  v12 = *(char **)(v11 - 24);
  v13 = *(char **)(v11 - 16);
  v14 = *(_QWORD *)(v11 - 8);
  *(_QWORD *)(v11 - 24) = *(_QWORD *)(a1 + 64);
  *(_QWORD *)(v11 - 16) = *(_QWORD *)(a1 + 72);
  *(_QWORD *)(v11 - 8) = *(_QWORD *)(a1 + 80);
  v15 = *(char **)(a1 + 136);
  result = *(char **)(a1 + 128);
  *(_QWORD *)(a1 + 64) = v12;
  *(_QWORD *)(a1 + 72) = v13;
  *(_QWORD *)(a1 + 80) = v14;
  if ( v15 != result )
  {
    v17 = v15 - 32;
    if ( a2 == *((_DWORD *)v15 - 8) )
    {
      v18 = (char *)*((_QWORD *)v17 + 2);
      v19 = (char *)*((_QWORD *)v17 + 1);
      if ( v18 == v19 )
        return result;
LABEL_13:
      v20 = v18 - v19;
      if ( v14 - (__int64)v13 >= (unsigned __int64)(v18 - v19) )
      {
        result = v19;
        v21 = &v13[v20];
        do
        {
          if ( v13 )
          {
            *(_QWORD *)v13 = *(_QWORD *)result;
            v22 = *((_QWORD *)result + 1);
            *((_QWORD *)v13 + 1) = v22;
            if ( v22 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd((volatile signed __int32 *)(v22 + 8), 1u);
              else
                ++*(_DWORD *)(v22 + 8);
            }
          }
          v13 += 16;
          result += 16;
        }
        while ( v21 != v13 );
        *(_QWORD *)(a1 + 72) += v20;
        return result;
      }
      v26 = v20 >> 4;
      v27 = 0x7FFFFFFFFFFFFFFLL;
      v28 = (v13 - v12) >> 4;
      if ( v26 > 0x7FFFFFFFFFFFFFFLL - v28 )
        sub_4262D8((__int64)"vector::_M_range_insert");
      if ( v26 < v28 )
        v26 = (v13 - v12) >> 4;
      v29 = __CFADD__(v28, v26);
      v30 = v28 + v26;
      if ( v29 )
      {
        v31 = 0x7FFFFFFFFFFFFFF0LL;
      }
      else
      {
        if ( !v30 )
        {
          v48 = 0;
          v33 = 0;
          goto LABEL_39;
        }
        if ( v30 <= 0x7FFFFFFFFFFFFFFLL )
          v27 = v30;
        v31 = 16 * v27;
      }
      v32 = sub_22077B0(v31);
      v12 = *(char **)(a1 + 64);
      v33 = (_QWORD *)v32;
      v48 = v31 + v32;
LABEL_39:
      if ( v13 == v12 )
      {
        v36 = v33;
      }
      else
      {
        v34 = v12;
        v35 = v33;
        v36 = (_QWORD *)((char *)v33 + v13 - v12);
        do
        {
          if ( v35 )
          {
            v37 = *(_QWORD *)v34;
            v35[1] = 0;
            *v35 = v37;
            v38 = *((_QWORD *)v34 + 1);
            *((_QWORD *)v34 + 1) = 0;
            v35[1] = v38;
            *(_QWORD *)v34 = 0;
          }
          v35 += 2;
          v34 += 16;
        }
        while ( v36 != v35 );
      }
      v39 = v19;
      v40 = (_QWORD *)((char *)v36 + v18 - v19);
      do
      {
        if ( v36 )
        {
          *v36 = *(_QWORD *)v39;
          v41 = *((_QWORD *)v39 + 1);
          v36[1] = v41;
          if ( v41 )
          {
            if ( &_pthread_key_create )
              _InterlockedAdd((volatile signed __int32 *)(v41 + 8), 1u);
            else
              ++*(_DWORD *)(v41 + 8);
          }
        }
        v36 += 2;
        v39 += 16;
      }
      while ( v36 != v40 );
      v42 = *(char **)(a1 + 72);
      if ( v13 == v42 )
      {
        v43 = v36;
      }
      else
      {
        v43 = (_QWORD *)((char *)v40 + v42 - v13);
        do
        {
          if ( v40 )
          {
            v44 = *(_QWORD *)v13;
            v40[1] = 0;
            *v40 = v44;
            v45 = *((_QWORD *)v13 + 1);
            *((_QWORD *)v13 + 1) = 0;
            v40[1] = v45;
            *(_QWORD *)v13 = 0;
          }
          v40 += 2;
          v13 += 16;
        }
        while ( v43 != v40 );
        v42 = *(char **)(a1 + 72);
      }
      v46 = *(_QWORD *)(a1 + 64);
      if ( (char *)v46 != v42 )
      {
        do
        {
          v47 = *(volatile signed __int32 **)(v46 + 8);
          if ( v47 )
            sub_A191D0(v47);
          v46 += 16;
        }
        while ( (char *)v46 != v42 );
        v42 = *(char **)(a1 + 64);
      }
      if ( v42 )
        j_j___libc_free_0(v42, *(_QWORD *)(a1 + 80) - (_QWORD)v42);
      *(_QWORD *)(a1 + 64) = v33;
      *(_QWORD *)(a1 + 72) = v43;
      *(_QWORD *)(a1 + 80) = v48;
      return (char *)v48;
    }
    while ( a2 != *(_DWORD *)result )
    {
      result += 32;
      if ( v15 == result )
        return result;
    }
    v18 = (char *)*((_QWORD *)result + 2);
    v19 = (char *)*((_QWORD *)result + 1);
    if ( v18 != v19 )
      goto LABEL_13;
  }
  return result;
}
