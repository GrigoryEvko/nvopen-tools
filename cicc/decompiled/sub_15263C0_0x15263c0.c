// Function: sub_15263C0
// Address: 0x15263c0
//
__int64 __fastcall sub_15263C0(__int64 **a1)
{
  __int64 *v2; // r8
  __int64 v3; // rcx
  __int64 v4; // rdi
  __int64 v5; // rsi
  char *v6; // rdx
  char *v7; // r12
  char *v8; // r13
  __int64 v9; // r14
  char *v10; // r15
  __int64 v11; // rcx
  __int64 *v12; // rax
  __int64 v13; // rcx
  volatile signed __int32 *v14; // rdi
  volatile signed __int32 *v15; // rdx
  volatile signed __int32 *v16; // rdi
  char *v17; // r14
  __int64 *v18; // r15
  __int64 result; // rax
  __int64 v20; // r14
  __int64 v21; // r12
  volatile signed __int32 *v22; // r13
  __int64 v23; // rax
  _QWORD *v24; // r13
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rcx
  char *v28; // r15
  char *v29; // r12
  volatile signed __int32 *v30; // rdi
  __int64 *v31; // r12
  int v32; // r13d
  __int64 v33; // rax
  __int64 v34; // rax
  volatile signed __int32 *v35; // rdi
  volatile signed __int32 *v36; // r15
  char *v37; // r12
  char *v38; // rdx
  __int64 v39; // rax
  volatile signed __int32 *v40; // [rsp+8h] [rbp-48h]
  __int64 v41; // [rsp+10h] [rbp-40h]
  __int64 v42; // [rsp+10h] [rbp-40h]
  __int64 *v43; // [rsp+18h] [rbp-38h]
  __int64 *v44; // [rsp+18h] [rbp-38h]
  char *v45; // [rsp+18h] [rbp-38h]
  __int64 *v46; // [rsp+18h] [rbp-38h]

  v43 = a1[7];
  sub_1524D80(a1, 0, *((_DWORD *)a1 + 4));
  v2 = v43;
  if ( *((_DWORD *)a1 + 2) )
  {
    v31 = *a1;
    v32 = *((_DWORD *)a1 + 3);
    v33 = *((unsigned int *)*a1 + 2);
    if ( (unsigned __int64)*((unsigned int *)*a1 + 3) - v33 <= 3 )
    {
      sub_16CD150(*a1, v31 + 2, v33 + 4, 1);
      v33 = *((unsigned int *)v31 + 2);
      v2 = v43;
    }
    *(_DWORD *)(*v31 + v33) = v32;
    *((_DWORD *)v31 + 2) += 4;
    a1[1] = 0;
  }
  v3 = *(v2 - 4);
  v4 = **a1;
  v5 = (unsigned int)(4 * v3);
  *(_DWORD *)(v4 + v5) = ~(_DWORD)v3 + (*((_DWORD *)*a1 + 2) >> 2);
  *((_DWORD *)a1 + 4) = *((_DWORD *)v2 - 10);
  if ( v2 - 3 != (__int64 *)(a1 + 3) )
  {
    v6 = (char *)*(v2 - 2);
    v7 = (char *)*(v2 - 3);
    v8 = (char *)a1[3];
    v9 = v6 - v7;
    if ( (char *)a1[5] - v8 < (unsigned __int64)(v6 - v7) )
    {
      if ( v9 )
      {
        if ( (unsigned __int64)v9 > 0x7FFFFFFFFFFFFFF0LL )
          sub_4261EA(v4, v5, v6);
        v45 = (char *)*(v2 - 2);
        v23 = sub_22077B0(v45 - v7);
        v6 = v45;
        v24 = (_QWORD *)v23;
      }
      else
      {
        v24 = 0;
      }
      if ( v6 != v7 )
      {
        v25 = v24;
        v26 = (_QWORD *)((char *)v24 + v6 - v7);
        do
        {
          if ( v25 )
          {
            *v25 = *(_QWORD *)v7;
            v27 = *((_QWORD *)v7 + 1);
            v25[1] = v27;
            if ( v27 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd((volatile signed __int32 *)(v27 + 8), 1u);
              else
                ++*(_DWORD *)(v27 + 8);
            }
          }
          v25 += 2;
          v7 += 16;
        }
        while ( v25 != v26 );
      }
      v28 = (char *)a1[4];
      v29 = (char *)a1[3];
      if ( v28 != v29 )
      {
        do
        {
          v30 = (volatile signed __int32 *)*((_QWORD *)v29 + 1);
          if ( v30 )
            sub_A191D0(v30);
          v29 += 16;
        }
        while ( v28 != v29 );
        v29 = (char *)a1[3];
      }
      if ( v29 )
        j_j___libc_free_0(v29, (char *)a1[5] - v29);
      v17 = (char *)v24 + v9;
      a1[3] = v24;
      a1[5] = (__int64 *)v17;
      goto LABEL_21;
    }
    v10 = (char *)a1[4];
    v11 = v10 - v8;
    if ( v9 > (unsigned __int64)(v10 - v8) )
    {
      v34 = v11 >> 4;
      if ( v11 > 0 )
      {
        do
        {
          v35 = (volatile signed __int32 *)*((_QWORD *)v8 + 1);
          *(_QWORD *)v8 = *(_QWORD *)v7;
          v36 = (volatile signed __int32 *)*((_QWORD *)v7 + 1);
          if ( v36 != v35 )
          {
            if ( v36 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd(v36 + 2, 1u);
              else
                ++*((_DWORD *)v36 + 2);
              v35 = (volatile signed __int32 *)*((_QWORD *)v8 + 1);
            }
            if ( v35 )
            {
              v42 = v34;
              v46 = v2;
              sub_A191D0(v35);
              v34 = v42;
              v2 = v46;
            }
            *((_QWORD *)v8 + 1) = v36;
          }
          v7 += 16;
          v8 += 16;
          --v34;
        }
        while ( v34 );
        v10 = (char *)a1[4];
        v8 = (char *)a1[3];
        v6 = (char *)*(v2 - 2);
        v7 = (char *)*(v2 - 3);
        v11 = v10 - v8;
      }
      v37 = &v7[v11];
      if ( v37 == v6 )
      {
        v17 = &v8[v9];
        goto LABEL_21;
      }
      v38 = &v10[v6 - v37];
      do
      {
        if ( v10 )
        {
          *(_QWORD *)v10 = *(_QWORD *)v37;
          v39 = *((_QWORD *)v37 + 1);
          *((_QWORD *)v10 + 1) = v39;
          if ( v39 )
          {
            if ( &_pthread_key_create )
              _InterlockedAdd((volatile signed __int32 *)(v39 + 8), 1u);
            else
              ++*(_DWORD *)(v39 + 8);
          }
        }
        v10 += 16;
        v37 += 16;
      }
      while ( v10 != v38 );
    }
    else
    {
      v12 = a1[3];
      v13 = v9 >> 4;
      if ( v9 <= 0 )
        goto LABEL_19;
      do
      {
        v14 = (volatile signed __int32 *)v12[1];
        *v12 = *(_QWORD *)v7;
        v15 = (volatile signed __int32 *)*((_QWORD *)v7 + 1);
        if ( v15 != v14 )
        {
          if ( v15 )
          {
            if ( &_pthread_key_create )
              _InterlockedAdd(v15 + 2, 1u);
            else
              ++*((_DWORD *)v15 + 2);
            v14 = (volatile signed __int32 *)v12[1];
          }
          if ( v14 )
          {
            v40 = v15;
            v41 = v13;
            v44 = v12;
            sub_A191D0(v14);
            v15 = v40;
            v13 = v41;
            v12 = v44;
          }
          v12[1] = (__int64)v15;
        }
        v7 += 16;
        v12 += 2;
        --v13;
      }
      while ( v13 );
      v8 += v9;
      while ( v10 != v8 )
      {
        v16 = (volatile signed __int32 *)*((_QWORD *)v8 + 1);
        if ( v16 )
          sub_A191D0(v16);
        v8 += 16;
LABEL_19:
        ;
      }
    }
    v17 = (char *)a1[3] + v9;
LABEL_21:
    a1[4] = (__int64 *)v17;
  }
  v18 = a1[7];
  result = (__int64)(v18 - 5);
  a1[7] = v18 - 5;
  v20 = *(v18 - 2);
  v21 = *(v18 - 3);
  if ( v20 != v21 )
  {
    do
    {
      while ( 1 )
      {
        v22 = *(volatile signed __int32 **)(v21 + 8);
        if ( v22 )
        {
          if ( &_pthread_key_create )
          {
            result = (unsigned int)_InterlockedExchangeAdd(v22 + 2, 0xFFFFFFFF);
          }
          else
          {
            result = *((unsigned int *)v22 + 2);
            *((_DWORD *)v22 + 2) = result - 1;
          }
          if ( (_DWORD)result == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 16LL))(v22);
            if ( &_pthread_key_create )
            {
              result = (unsigned int)_InterlockedExchangeAdd(v22 + 3, 0xFFFFFFFF);
            }
            else
            {
              result = *((unsigned int *)v22 + 3);
              *((_DWORD *)v22 + 3) = result - 1;
            }
            if ( (_DWORD)result == 1 )
              break;
          }
        }
        v21 += 16;
        if ( v20 == v21 )
          goto LABEL_33;
      }
      v21 += 16;
      result = (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 24LL))(v22);
    }
    while ( v20 != v21 );
LABEL_33:
    v21 = *(v18 - 3);
  }
  if ( v21 )
    return j_j___libc_free_0(v21, *(v18 - 1) - v21);
  return result;
}
