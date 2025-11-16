// Function: sub_B428A0
// Address: 0xb428a0
//
__int64 *__fastcall sub_B428A0(__int64 *a1, _BYTE *a2, __int64 a3)
{
  unsigned int v4; // edi
  __int64 *v5; // r13
  char v6; // dl
  unsigned __int64 v7; // rsi
  _BYTE *v8; // rax
  _BYTE *v9; // r12
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  char *v12; // r14
  int v13; // eax
  __int64 v14; // rdx
  bool v15; // zf
  _QWORD *v16; // rcx
  _QWORD *v17; // rbx
  __int64 v18; // rax
  _BYTE *v19; // rbx
  _BYTE *v20; // r12
  __int64 v21; // r14
  _QWORD *v22; // r15
  _QWORD *v23; // r14
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  int v27; // r14d
  char *v28; // r15
  int v29; // ebx
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // rbx
  __int64 v34; // r13
  _QWORD *v35; // r15
  _QWORD *v36; // r13
  _QWORD *v37; // r12
  _QWORD *v38; // rbx
  _BYTE *v39; // rbx
  _BYTE *v40; // r12
  __int64 v41; // r14
  _QWORD *v42; // r15
  _QWORD *v43; // r14
  _QWORD *v44; // rbx
  _QWORD *v45; // r12
  __int64 v46; // r15
  __int64 v47; // rdx
  __int64 v48; // r14
  __int64 v49; // r12
  __int64 v50; // rbx
  _QWORD *v51; // r13
  _QWORD *v52; // rbx
  _QWORD *v53; // r12
  _QWORD *v54; // rbx
  pthread_mutex_t *mutex; // [rsp+0h] [rbp-150h]
  _QWORD *v56; // [rsp+8h] [rbp-148h]
  __int64 v58; // [rsp+40h] [rbp-110h]
  __int64 *v59; // [rsp+40h] [rbp-110h]
  _BYTE *v60; // [rsp+48h] [rbp-108h]
  __int64 v61; // [rsp+48h] [rbp-108h]
  __int64 v62; // [rsp+48h] [rbp-108h]
  __int64 v63; // [rsp+58h] [rbp-F8h] BYREF
  _QWORD v64[2]; // [rsp+60h] [rbp-F0h] BYREF
  _BYTE *v65; // [rsp+70h] [rbp-E0h]
  __int64 v66; // [rsp+78h] [rbp-D8h]
  _BYTE v67[32]; // [rsp+80h] [rbp-D0h] BYREF
  _BYTE *v68; // [rsp+A0h] [rbp-B0h]
  __int64 v69; // [rsp+A8h] [rbp-A8h]
  _BYTE v70[160]; // [rsp+B0h] [rbp-A0h] BYREF

  v60 = a2;
  if ( !::mutex )
    sub_C7D570(&::mutex, sub_B3AE10, sub_B3AE40);
  mutex = ::mutex;
  if ( &_pthread_key_create )
  {
    v4 = pthread_mutex_lock(::mutex);
    if ( v4 )
      sub_4264C5(v4);
  }
  if ( !qword_4F817F0 )
    sub_C7D570(&qword_4F817F0, sub_B3B280, sub_B3B380);
  v5 = (__int64 *)sub_B41FC0(qword_4F817F0, a2, a3);
  if ( !v6 )
    goto LABEL_41;
  v58 = (__int64)&a2[a3];
  if ( a2 == &a2[a3] )
    goto LABEL_41;
  v56 = v5 + 2;
  while ( 1 )
  {
    v64[1] = 0;
    v64[0] = 0xFFFFFFFF00000000LL;
    v7 = v58;
    LOBYTE(v63) = 44;
    v65 = v67;
    v66 = 0x100000000LL;
    v68 = v70;
    v69 = 0x200000000LL;
    v8 = sub_B3AF10(v60, v58, (char *)&v63);
    v9 = v8;
    if ( v8 == v60
      || (v7 = (unsigned __int64)v60, (unsigned __int8)sub_B3CA00((__int64)v64, v60, v8 - v60, (__int64)v5)) )
    {
      v61 = *v5;
      v30 = *v5 + 192LL * *((unsigned int *)v5 + 2);
      if ( *v5 == v30 )
        goto LABEL_78;
      v59 = v5;
      do
      {
        v31 = *(unsigned int *)(v30 - 120);
        v32 = *(_QWORD *)(v30 - 128);
        v30 -= 192;
        v33 = v32 + 56 * v31;
        if ( v32 != v33 )
        {
          do
          {
            v34 = *(unsigned int *)(v33 - 40);
            v35 = *(_QWORD **)(v33 - 48);
            v33 -= 56;
            v36 = &v35[4 * v34];
            if ( v35 != v36 )
            {
              do
              {
                v36 -= 4;
                if ( (_QWORD *)*v36 != v36 + 2 )
                {
                  v7 = v36[2] + 1LL;
                  j_j___libc_free_0(*v36, v7);
                }
              }
              while ( v35 != v36 );
              v35 = *(_QWORD **)(v33 + 8);
            }
            if ( v35 != (_QWORD *)(v33 + 24) )
              _libc_free(v35, v7);
          }
          while ( v32 != v33 );
          v32 = *(_QWORD *)(v30 + 64);
        }
        if ( v32 != v30 + 80 )
          _libc_free(v32, v7);
        v37 = *(_QWORD **)(v30 + 16);
        v38 = &v37[4 * *(unsigned int *)(v30 + 24)];
        if ( v37 != v38 )
        {
          do
          {
            v38 -= 4;
            if ( (_QWORD *)*v38 != v38 + 2 )
            {
              v7 = v38[2] + 1LL;
              j_j___libc_free_0(*v38, v7);
            }
          }
          while ( v37 != v38 );
          v37 = *(_QWORD **)(v30 + 16);
        }
        if ( v37 != (_QWORD *)(v30 + 32) )
          _libc_free(v37, v7);
      }
      while ( v61 != v30 );
      goto LABEL_77;
    }
    v10 = *((unsigned int *)v5 + 2);
    v7 = *((unsigned int *)v5 + 3);
    v11 = *v5;
    v12 = (char *)v64;
    v13 = *((_DWORD *)v5 + 2);
    if ( v10 + 1 > v7 )
    {
      if ( v11 > (unsigned __int64)v64 || (unsigned __int64)v64 >= v11 + 192 * v10 )
      {
        v7 = sub_C8D7D0(v5, v56, v10 + 1, 192, &v63);
        sub_B3DE10(v5, v7);
        v27 = v63;
        if ( (_QWORD *)*v5 != v56 )
          _libc_free(*v5, v7);
        v10 = *((unsigned int *)v5 + 2);
        *((_DWORD *)v5 + 3) = v27;
        v11 = v7;
        *v5 = v7;
        v12 = (char *)v64;
        v13 = v10;
      }
      else
      {
        v28 = (char *)v64 - v11;
        v7 = sub_C8D7D0(v5, v56, v10 + 1, 192, &v63);
        sub_B3DE10(v5, v7);
        v29 = v63;
        if ( (_QWORD *)*v5 != v56 )
          _libc_free(*v5, v7);
        *v5 = v7;
        v10 = *((unsigned int *)v5 + 2);
        v11 = *v5;
        *((_DWORD *)v5 + 3) = v29;
        v13 = v10;
        v12 = &v28[v11];
      }
    }
    v14 = 192 * v10;
    v15 = v14 + v11 == 0;
    v16 = (_QWORD *)(v14 + v11);
    v17 = v16;
    if ( !v15 )
    {
      *v16 = *(_QWORD *)v12;
      v16[1] = *((_QWORD *)v12 + 1);
      v16[2] = v16 + 4;
      v16[3] = 0x100000000LL;
      v7 = *((unsigned int *)v12 + 6);
      if ( (_DWORD)v7 )
      {
        v7 = (unsigned __int64)(v12 + 16);
        sub_B3C2C0((__int64)(v16 + 2), (__int64 *)v12 + 2);
      }
      v17[8] = v17 + 10;
      v17[9] = 0x200000000LL;
      if ( *((_DWORD *)v12 + 18) )
      {
        v7 = (unsigned __int64)(v12 + 64);
        sub_B3DB20((__int64)(v17 + 8), (__int64)(v12 + 64));
      }
      v13 = *((_DWORD *)v5 + 2);
    }
    v18 = (unsigned int)(v13 + 1);
    *((_DWORD *)v5 + 2) = v18;
    if ( (_BYTE *)v58 == v9 )
    {
      v60 = (_BYTE *)v58;
      goto LABEL_21;
    }
    v60 = v9 + 1;
    if ( v9 + 1 == (_BYTE *)v58 )
      break;
LABEL_21:
    v19 = v68;
    v20 = &v68[56 * (unsigned int)v69];
    if ( v68 != v20 )
    {
      do
      {
        v21 = *((unsigned int *)v20 - 10);
        v22 = (_QWORD *)*((_QWORD *)v20 - 6);
        v20 -= 56;
        v23 = &v22[4 * v21];
        if ( v22 != v23 )
        {
          do
          {
            v23 -= 4;
            if ( (_QWORD *)*v23 != v23 + 2 )
            {
              v7 = v23[2] + 1LL;
              j_j___libc_free_0(*v23, v7);
            }
          }
          while ( v22 != v23 );
          v22 = (_QWORD *)*((_QWORD *)v20 + 1);
        }
        if ( v22 != (_QWORD *)(v20 + 24) )
          _libc_free(v22, v7);
      }
      while ( v19 != v20 );
      v20 = v68;
    }
    if ( v20 != v70 )
      _libc_free(v20, v7);
    v24 = v65;
    v25 = &v65[32 * (unsigned int)v66];
    if ( v65 != (_BYTE *)v25 )
    {
      do
      {
        v25 -= 4;
        if ( (_QWORD *)*v25 != v25 + 2 )
        {
          v7 = v25[2] + 1LL;
          j_j___libc_free_0(*v25, v7);
        }
      }
      while ( v24 != v25 );
      v25 = v65;
    }
    if ( v25 != (_QWORD *)v67 )
      _libc_free(v25, v7);
    if ( v60 == (_BYTE *)v58 )
      goto LABEL_41;
  }
  v62 = *v5;
  if ( *v5 != *v5 + 192 * v18 )
  {
    v59 = v5;
    v46 = *v5 + 192 * v18;
    do
    {
      v47 = *(unsigned int *)(v46 - 120);
      v48 = *(_QWORD *)(v46 - 128);
      v46 -= 192;
      v49 = v48 + 56 * v47;
      if ( v48 != v49 )
      {
        do
        {
          v50 = *(unsigned int *)(v49 - 40);
          v51 = *(_QWORD **)(v49 - 48);
          v49 -= 56;
          v52 = &v51[4 * v50];
          if ( v51 != v52 )
          {
            do
            {
              v52 -= 4;
              if ( (_QWORD *)*v52 != v52 + 2 )
              {
                v7 = v52[2] + 1LL;
                j_j___libc_free_0(*v52, v7);
              }
            }
            while ( v51 != v52 );
            v51 = *(_QWORD **)(v49 + 8);
          }
          if ( v51 != (_QWORD *)(v49 + 24) )
            _libc_free(v51, v7);
        }
        while ( v48 != v49 );
        v48 = *(_QWORD *)(v46 + 64);
      }
      if ( v48 != v46 + 80 )
        _libc_free(v48, v7);
      v53 = *(_QWORD **)(v46 + 16);
      v54 = &v53[4 * *(unsigned int *)(v46 + 24)];
      if ( v53 != v54 )
      {
        do
        {
          v54 -= 4;
          if ( (_QWORD *)*v54 != v54 + 2 )
          {
            v7 = v54[2] + 1LL;
            j_j___libc_free_0(*v54, v7);
          }
        }
        while ( v53 != v54 );
        v53 = *(_QWORD **)(v46 + 16);
      }
      if ( v53 != (_QWORD *)(v46 + 32) )
        _libc_free(v53, v7);
    }
    while ( v62 != v46 );
LABEL_77:
    v5 = v59;
  }
LABEL_78:
  *((_DWORD *)v5 + 2) = 0;
  v39 = v68;
  v40 = &v68[56 * (unsigned int)v69];
  if ( v68 != v40 )
  {
    do
    {
      v41 = *((unsigned int *)v40 - 10);
      v42 = (_QWORD *)*((_QWORD *)v40 - 6);
      v40 -= 56;
      v43 = &v42[4 * v41];
      if ( v42 != v43 )
      {
        do
        {
          v43 -= 4;
          if ( (_QWORD *)*v43 != v43 + 2 )
          {
            v7 = v43[2] + 1LL;
            j_j___libc_free_0(*v43, v7);
          }
        }
        while ( v42 != v43 );
        v42 = (_QWORD *)*((_QWORD *)v40 + 1);
      }
      if ( v42 != (_QWORD *)(v40 + 24) )
        _libc_free(v42, v7);
    }
    while ( v39 != v40 );
    v40 = v68;
  }
  if ( v40 != v70 )
    _libc_free(v40, v7);
  v44 = v65;
  v45 = &v65[32 * (unsigned int)v66];
  if ( v65 != (_BYTE *)v45 )
  {
    do
    {
      v45 -= 4;
      if ( (_QWORD *)*v45 != v45 + 2 )
      {
        v7 = v45[2] + 1LL;
        j_j___libc_free_0(*v45, v7);
      }
    }
    while ( v44 != v45 );
    v45 = v65;
  }
  if ( v45 != (_QWORD *)v67 )
    _libc_free(v45, v7);
LABEL_41:
  *a1 = (__int64)(a1 + 2);
  a1[1] = 0x400000000LL;
  if ( *((_DWORD *)v5 + 2) )
    sub_B3E940(a1, (__int64)v5);
  if ( &_pthread_key_create )
    pthread_mutex_unlock(mutex);
  return a1;
}
