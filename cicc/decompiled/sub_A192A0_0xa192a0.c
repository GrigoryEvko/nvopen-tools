// Function: sub_A192A0
// Address: 0xa192a0
//
__int64 __fastcall sub_A192A0(__int64 a1)
{
  __int64 v2; // r15
  _QWORD *v3; // r13
  unsigned __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  int v8; // edx
  __int64 v9; // rsi
  char *v10; // rcx
  char *v11; // r12
  char *v12; // r13
  __int64 v13; // r14
  char *v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // r15
  __int64 v17; // rcx
  volatile signed __int32 *v18; // rdi
  volatile signed __int32 *v19; // rdx
  volatile signed __int32 *v20; // rdi
  char *v21; // r14
  __int64 v22; // r14
  __int64 result; // rax
  __int64 v24; // r15
  __int64 v25; // r12
  volatile signed __int32 *v26; // r13
  __int64 v27; // rdi
  __int64 v28; // rax
  _QWORD *v29; // r13
  _QWORD *v30; // rax
  _QWORD *v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // r15
  __int64 v34; // r12
  volatile signed __int32 *v35; // rdi
  _QWORD *v36; // r12
  int v37; // r13d
  __int64 v38; // rax
  __int64 v39; // rdx
  volatile signed __int32 *v40; // rdi
  volatile signed __int32 *v41; // rax
  char *v42; // r12
  char *v43; // rcx
  __int64 v44; // rdx
  unsigned __int64 v45; // rdx
  volatile signed __int32 *v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+10h] [rbp-40h]
  volatile signed __int32 *v48; // [rsp+10h] [rbp-40h]
  char *v49; // [rsp+18h] [rbp-38h]
  char *v50; // [rsp+18h] [rbp-38h]
  char *v51; // [rsp+18h] [rbp-38h]
  __int64 v52; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 112);
  sub_A17B10(a1, 0, *(_DWORD *)(a1 + 56));
  if ( *(_DWORD *)(a1 + 48) )
  {
    v36 = *(_QWORD **)(a1 + 24);
    v37 = *(_DWORD *)(a1 + 52);
    v38 = v36[1];
    if ( (unsigned __int64)(v38 + 4) > v36[2] )
    {
      sub_C8D290(*(_QWORD *)(a1 + 24), v36 + 3, v38 + 4, 1);
      v38 = v36[1];
    }
    *(_DWORD *)(*v36 + v38) = v37;
    v36[1] += 4LL;
    *(_QWORD *)(a1 + 48) = 0;
  }
  v3 = *(_QWORD **)(a1 + 32);
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
  if ( v3 && (unsigned __int8)sub_CB7440(*(_QWORD *)(a1 + 32)) )
  {
    if ( !(unsigned __int8)sub_CB7440(v3) )
      BUG();
    v4 += (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 80LL))(v3) + v3[4] - v3[2];
  }
  v5 = *(_QWORD *)(v2 - 32);
  v6 = 32 * v5;
  v7 = ~v5 + (v4 >> 2);
  sub_A177B0(a1, 32 * v5, (unsigned __int8)v7);
  v8 = BYTE1(v7);
  LODWORD(v7) = WORD1(v7);
  sub_A177B0(a1, v6 + 8, v8);
  sub_A177B0(a1, v6 + 16, (unsigned __int8)v7);
  v9 = v6 + 24;
  sub_A177B0(a1, v6 + 24, (unsigned int)v7 >> 8);
  *(_DWORD *)(a1 + 56) = *(_DWORD *)(v2 - 40);
  if ( v2 - 24 != a1 + 64 )
  {
    v10 = *(char **)(v2 - 16);
    v11 = *(char **)(v2 - 24);
    v12 = *(char **)(a1 + 64);
    v13 = v10 - v11;
    if ( *(_QWORD *)(a1 + 80) - (_QWORD)v12 < (unsigned __int64)(v10 - v11) )
    {
      if ( v13 )
      {
        if ( (unsigned __int64)v13 > 0x7FFFFFFFFFFFFFF0LL )
          sub_4261EA(a1, v9, v2 - 24, v10);
        v51 = *(char **)(v2 - 16);
        v28 = sub_22077B0(v51 - v11);
        v10 = v51;
        v29 = (_QWORD *)v28;
      }
      else
      {
        v29 = 0;
      }
      if ( v10 != v11 )
      {
        v30 = v29;
        v31 = (_QWORD *)((char *)v29 + v10 - v11);
        do
        {
          if ( v30 )
          {
            *v30 = *(_QWORD *)v11;
            v32 = *((_QWORD *)v11 + 1);
            v30[1] = v32;
            if ( v32 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd((volatile signed __int32 *)(v32 + 8), 1u);
              else
                ++*(_DWORD *)(v32 + 8);
            }
          }
          v30 += 2;
          v11 += 16;
        }
        while ( v30 != v31 );
      }
      v33 = *(_QWORD *)(a1 + 72);
      v34 = *(_QWORD *)(a1 + 64);
      if ( v33 != v34 )
      {
        do
        {
          v35 = *(volatile signed __int32 **)(v34 + 8);
          if ( v35 )
            sub_A191D0(v35);
          v34 += 16;
        }
        while ( v33 != v34 );
        v34 = *(_QWORD *)(a1 + 64);
      }
      if ( v34 )
        j_j___libc_free_0(v34, *(_QWORD *)(a1 + 80) - v34);
      v21 = (char *)v29 + v13;
      *(_QWORD *)(a1 + 64) = v29;
      *(_QWORD *)(a1 + 80) = v21;
      goto LABEL_25;
    }
    v14 = *(char **)(a1 + 72);
    v15 = v14 - v12;
    if ( v13 > (unsigned __int64)(v14 - v12) )
    {
      v39 = v15 >> 4;
      if ( v15 > 0 )
      {
        do
        {
          v40 = (volatile signed __int32 *)*((_QWORD *)v12 + 1);
          *(_QWORD *)v12 = *(_QWORD *)v11;
          v41 = (volatile signed __int32 *)*((_QWORD *)v11 + 1);
          if ( v41 != v40 )
          {
            if ( v41 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd(v41 + 2, 1u);
              else
                ++*((_DWORD *)v41 + 2);
              v40 = (volatile signed __int32 *)*((_QWORD *)v12 + 1);
            }
            if ( v40 )
            {
              v48 = v41;
              v52 = v39;
              sub_A191D0(v40);
              v41 = v48;
              v39 = v52;
            }
            *((_QWORD *)v12 + 1) = v41;
          }
          v11 += 16;
          v12 += 16;
          --v39;
        }
        while ( v39 );
        v14 = *(char **)(a1 + 72);
        v12 = *(char **)(a1 + 64);
        v10 = *(char **)(v2 - 16);
        v11 = *(char **)(v2 - 24);
        v15 = v14 - v12;
      }
      v42 = &v11[v15];
      if ( v42 == v10 )
      {
        v21 = &v12[v13];
        goto LABEL_25;
      }
      v43 = &v14[v10 - v42];
      do
      {
        if ( v14 )
        {
          *(_QWORD *)v14 = *(_QWORD *)v42;
          v44 = *((_QWORD *)v42 + 1);
          *((_QWORD *)v14 + 1) = v44;
          if ( v44 )
          {
            if ( &_pthread_key_create )
              _InterlockedAdd((volatile signed __int32 *)(v44 + 8), 1u);
            else
              ++*(_DWORD *)(v44 + 8);
          }
        }
        v14 += 16;
        v42 += 16;
      }
      while ( v14 != v43 );
    }
    else
    {
      v16 = *(_QWORD **)(a1 + 64);
      v17 = v13 >> 4;
      if ( v13 <= 0 )
        goto LABEL_23;
      do
      {
        v18 = (volatile signed __int32 *)v16[1];
        *v16 = *(_QWORD *)v11;
        v19 = (volatile signed __int32 *)*((_QWORD *)v11 + 1);
        if ( v19 != v18 )
        {
          if ( v19 )
          {
            if ( &_pthread_key_create )
              _InterlockedAdd(v19 + 2, 1u);
            else
              ++*((_DWORD *)v19 + 2);
            v18 = (volatile signed __int32 *)v16[1];
          }
          if ( v18 )
          {
            v46 = v19;
            v47 = v17;
            v49 = v14;
            sub_A191D0(v18);
            v19 = v46;
            v17 = v47;
            v14 = v49;
          }
          v16[1] = v19;
        }
        v11 += 16;
        v16 += 2;
        --v17;
      }
      while ( v17 );
      v12 += v13;
      while ( v14 != v12 )
      {
        v20 = (volatile signed __int32 *)*((_QWORD *)v12 + 1);
        if ( v20 )
        {
          v50 = v14;
          sub_A191D0(v20);
          v14 = v50;
        }
        v12 += 16;
LABEL_23:
        ;
      }
    }
    v21 = (char *)(*(_QWORD *)(a1 + 64) + v13);
LABEL_25:
    *(_QWORD *)(a1 + 72) = v21;
  }
  v22 = *(_QWORD *)(a1 + 112);
  result = v22 - 40;
  *(_QWORD *)(a1 + 112) = v22 - 40;
  v24 = *(_QWORD *)(v22 - 16);
  v25 = *(_QWORD *)(v22 - 24);
  if ( v24 != v25 )
  {
    do
    {
      while ( 1 )
      {
        v26 = *(volatile signed __int32 **)(v25 + 8);
        if ( v26 )
        {
          if ( &_pthread_key_create )
          {
            result = (unsigned int)_InterlockedExchangeAdd(v26 + 2, 0xFFFFFFFF);
          }
          else
          {
            result = *((unsigned int *)v26 + 2);
            *((_DWORD *)v26 + 2) = result - 1;
          }
          if ( (_DWORD)result == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v26 + 16LL))(v26);
            if ( &_pthread_key_create )
            {
              result = (unsigned int)_InterlockedExchangeAdd(v26 + 3, 0xFFFFFFFF);
            }
            else
            {
              result = *((unsigned int *)v26 + 3);
              *((_DWORD *)v26 + 3) = result - 1;
            }
            if ( (_DWORD)result == 1 )
              break;
          }
        }
        v25 += 16;
        if ( v24 == v25 )
          goto LABEL_37;
      }
      v25 += 16;
      result = (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v26 + 24LL))(v26);
    }
    while ( v24 != v25 );
LABEL_37:
    v25 = *(_QWORD *)(v22 - 24);
  }
  if ( v25 )
    result = j_j___libc_free_0(v25, *(_QWORD *)(v22 - 8) - v25);
  v27 = *(_QWORD *)(a1 + 32);
  if ( v27 )
  {
    result = *(_QWORD *)(a1 + 24);
    if ( *(_QWORD *)(result + 8) )
    {
      if ( !*(_BYTE *)(a1 + 96) )
      {
        result = sub_CB7440(v27);
        if ( (_BYTE)result )
        {
          result = *(_QWORD *)(a1 + 24);
          v45 = *(_QWORD *)(result + 8);
          if ( v45 > *(_QWORD *)(a1 + 40) )
          {
            sub_CB6200(*(_QWORD *)(a1 + 32), *(_QWORD *)result, v45);
            result = *(_QWORD *)(a1 + 24);
            *(_QWORD *)(result + 8) = 0;
          }
        }
      }
    }
  }
  return result;
}
