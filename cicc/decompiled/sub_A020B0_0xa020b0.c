// Function: sub_A020B0
// Address: 0xa020b0
//
void __fastcall sub_A020B0(__int64 a1, char **a2, __int64 a3, __int64 a4)
{
  char *v5; // r14
  char *v6; // rbx
  char *v7; // r12
  __int64 v8; // rdx
  char *v9; // r13
  __int64 v10; // rax
  char *v11; // r14
  __int64 v12; // rsi
  volatile signed __int32 *v13; // rdi
  volatile signed __int32 *v14; // rcx
  signed __int32 v15; // r8d
  void (*v16)(void); // r8
  signed __int32 v17; // eax
  void (*v18)(void); // r8
  void (*v19)(); // rcx
  signed __int32 v20; // eax
  volatile signed __int32 *v21; // r14
  signed __int32 v22; // ecx
  __int64 v23; // rax
  _QWORD *v24; // r12
  _QWORD *v25; // rax
  _QWORD *v26; // r14
  __int64 v27; // rcx
  char *v28; // rbx
  char *v29; // r13
  volatile signed __int32 *v30; // r14
  signed __int32 v31; // ecx
  void (*v32)(); // rcx
  signed __int32 v33; // eax
  __int64 (__fastcall *v34)(__int64); // rcx
  char *v35; // rbx
  char *v36; // r14
  __int64 v37; // rax
  char *v38; // rdx
  __int64 v39; // rcx
  volatile signed __int32 *v40; // r13
  volatile signed __int32 *v41; // r14
  signed __int32 v42; // edi
  void (*v43)(); // r8
  signed __int32 v44; // eax
  __int64 (__fastcall *v45)(__int64); // r8
  __int64 (__fastcall *v46)(__int64); // rcx
  volatile signed __int32 *v47; // [rsp-58h] [rbp-58h]
  volatile signed __int32 *v48; // [rsp-50h] [rbp-50h]
  __int64 v49; // [rsp-48h] [rbp-48h]
  __int64 v50; // [rsp-48h] [rbp-48h]
  __int64 v51; // [rsp-48h] [rbp-48h]
  __int64 v52; // [rsp-40h] [rbp-40h]
  __int64 v53; // [rsp-40h] [rbp-40h]
  __int64 v54; // [rsp-40h] [rbp-40h]
  __int64 v55; // [rsp-40h] [rbp-40h]
  __int64 v56; // [rsp-40h] [rbp-40h]
  __int64 v57; // [rsp-40h] [rbp-40h]
  __int64 v58; // [rsp-40h] [rbp-40h]
  __int64 v59; // [rsp-40h] [rbp-40h]
  __int64 v60; // [rsp-40h] [rbp-40h]

  if ( a2 != (char **)a1 )
  {
    v5 = a2[1];
    v6 = *a2;
    v7 = *(char **)a1;
    v8 = v5 - *a2;
    if ( *(_QWORD *)(a1 + 16) - *(_QWORD *)a1 < (unsigned __int64)v8 )
    {
      if ( v8 )
      {
        if ( (unsigned __int64)v8 > 0x7FFFFFFFFFFFFFF0LL )
          sub_4261EA(a1, a2, v8, a4);
        v53 = a2[1] - *a2;
        v23 = sub_22077B0(v53);
        v8 = v53;
        v24 = (_QWORD *)v23;
      }
      else
      {
        v24 = 0;
      }
      if ( v5 != v6 )
      {
        v25 = v24;
        v26 = (_QWORD *)((char *)v24 + v5 - v6);
        do
        {
          if ( v25 )
          {
            *v25 = *(_QWORD *)v6;
            v27 = *((_QWORD *)v6 + 1);
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
          v6 += 16;
        }
        while ( v25 != v26 );
      }
      v28 = *(char **)(a1 + 8);
      v29 = *(char **)a1;
      if ( v28 != *(char **)a1 )
      {
        do
        {
          v30 = (volatile signed __int32 *)*((_QWORD *)v29 + 1);
          if ( v30 )
          {
            if ( &_pthread_key_create )
            {
              v31 = _InterlockedExchangeAdd(v30 + 2, 0xFFFFFFFF);
            }
            else
            {
              v31 = *((_DWORD *)v30 + 2);
              *((_DWORD *)v30 + 2) = v31 - 1;
            }
            if ( v31 == 1 )
            {
              v32 = *(void (**)())(*(_QWORD *)v30 + 16LL);
              if ( v32 != nullsub_25 )
              {
                v59 = v8;
                ((void (__fastcall *)(volatile signed __int32 *))v32)(v30);
                v8 = v59;
              }
              if ( &_pthread_key_create )
              {
                v33 = _InterlockedExchangeAdd(v30 + 3, 0xFFFFFFFF);
              }
              else
              {
                v33 = *((_DWORD *)v30 + 3);
                *((_DWORD *)v30 + 3) = v33 - 1;
              }
              if ( v33 == 1 )
              {
                v54 = v8;
                v34 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v30 + 24LL);
                if ( v34 == sub_9C26E0 )
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v30 + 8LL))(v30);
                else
                  v34((__int64)v30);
                v8 = v54;
              }
            }
          }
          v29 += 16;
        }
        while ( v28 != v29 );
        v29 = *(char **)a1;
      }
      if ( v29 )
      {
        v57 = v8;
        j_j___libc_free_0(v29, *(_QWORD *)(a1 + 16) - (_QWORD)v29);
        v8 = v57;
      }
      v38 = (char *)v24 + v8;
      *(_QWORD *)a1 = v24;
      *(_QWORD *)(a1 + 16) = v38;
      goto LABEL_70;
    }
    v9 = *(char **)(a1 + 8);
    v10 = v9 - v7;
    if ( v8 > (unsigned __int64)(v9 - v7) )
    {
      v39 = v10 >> 4;
      if ( v10 > 0 )
      {
        do
        {
          v40 = (volatile signed __int32 *)*((_QWORD *)v7 + 1);
          *(_QWORD *)v7 = *(_QWORD *)v6;
          v41 = (volatile signed __int32 *)*((_QWORD *)v6 + 1);
          if ( v41 != v40 )
          {
            if ( v41 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd(v41 + 2, 1u);
              else
                ++*((_DWORD *)v41 + 2);
              v40 = (volatile signed __int32 *)*((_QWORD *)v7 + 1);
            }
            if ( v40 )
            {
              if ( &_pthread_key_create )
              {
                v42 = _InterlockedExchangeAdd(v40 + 2, 0xFFFFFFFF);
              }
              else
              {
                v42 = *((_DWORD *)v40 + 2);
                *((_DWORD *)v40 + 2) = v42 - 1;
              }
              if ( v42 == 1 )
              {
                v43 = *(void (**)())(*(_QWORD *)v40 + 16LL);
                if ( v43 != nullsub_25 )
                {
                  v50 = v39;
                  v60 = v8;
                  ((void (__fastcall *)(volatile signed __int32 *))v43)(v40);
                  v39 = v50;
                  v8 = v60;
                }
                if ( &_pthread_key_create )
                {
                  v44 = _InterlockedExchangeAdd(v40 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v44 = *((_DWORD *)v40 + 3);
                  *((_DWORD *)v40 + 3) = v44 - 1;
                }
                if ( v44 == 1 )
                {
                  v49 = v39;
                  v45 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v40 + 24LL);
                  v55 = v8;
                  if ( v45 == sub_9C26E0 )
                  {
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v40 + 8LL))(v40);
                    v8 = v55;
                    v39 = v49;
                  }
                  else
                  {
                    v45((__int64)v40);
                    v39 = v49;
                    v8 = v55;
                  }
                }
              }
            }
            *((_QWORD *)v7 + 1) = v41;
          }
          v6 += 16;
          v7 += 16;
          --v39;
        }
        while ( v39 );
        v9 = *(char **)(a1 + 8);
        v7 = *(char **)a1;
        v5 = a2[1];
        v6 = *a2;
        v10 = (__int64)&v9[-*(_QWORD *)a1];
      }
      v35 = &v6[v10];
      if ( v35 == v5 )
      {
        v38 = &v7[v8];
        goto LABEL_70;
      }
      v36 = &v9[v5 - v35];
      do
      {
        if ( v9 )
        {
          *(_QWORD *)v9 = *(_QWORD *)v35;
          v37 = *((_QWORD *)v35 + 1);
          *((_QWORD *)v9 + 1) = v37;
          if ( v37 )
          {
            if ( &_pthread_key_create )
              _InterlockedAdd((volatile signed __int32 *)(v37 + 8), 1u);
            else
              ++*(_DWORD *)(v37 + 8);
          }
        }
        v9 += 16;
        v35 += 16;
      }
      while ( v9 != v36 );
    }
    else
    {
      v11 = *(char **)a1;
      v12 = v8 >> 4;
      if ( v8 > 0 )
      {
        do
        {
          v13 = (volatile signed __int32 *)*((_QWORD *)v11 + 1);
          *(_QWORD *)v11 = *(_QWORD *)v6;
          v14 = (volatile signed __int32 *)*((_QWORD *)v6 + 1);
          if ( v14 != v13 )
          {
            if ( v14 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd(v14 + 2, 1u);
              else
                ++*((_DWORD *)v14 + 2);
              v13 = (volatile signed __int32 *)*((_QWORD *)v11 + 1);
            }
            if ( v13 )
            {
              if ( &_pthread_key_create )
              {
                v15 = _InterlockedExchangeAdd(v13 + 2, 0xFFFFFFFF);
              }
              else
              {
                v15 = *((_DWORD *)v13 + 2);
                *((_DWORD *)v13 + 2) = v15 - 1;
              }
              if ( v15 == 1 )
              {
                v16 = *(void (**)(void))(*(_QWORD *)v13 + 16LL);
                if ( v16 != nullsub_25 )
                {
                  v47 = v14;
                  v51 = v8;
                  v16();
                  v14 = v47;
                  v8 = v51;
                }
                if ( &_pthread_key_create )
                {
                  v17 = _InterlockedExchangeAdd(v13 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v17 = *((_DWORD *)v13 + 3);
                  *((_DWORD *)v13 + 3) = v17 - 1;
                }
                if ( v17 == 1 )
                {
                  v48 = v14;
                  v18 = *(void (**)(void))(*(_QWORD *)v13 + 24LL);
                  v52 = v8;
                  if ( (char *)v18 == (char *)sub_9C26E0 )
                  {
                    (*(void (**)(void))(*(_QWORD *)v13 + 8LL))();
                    v8 = v52;
                    v14 = v48;
                  }
                  else
                  {
                    v18();
                    v14 = v48;
                    v8 = v52;
                  }
                }
              }
            }
            *((_QWORD *)v11 + 1) = v14;
          }
          v6 += 16;
          v11 += 16;
          --v12;
        }
        while ( v12 );
        v7 += v8;
      }
      while ( v9 != v7 )
      {
        v21 = (volatile signed __int32 *)*((_QWORD *)v7 + 1);
        if ( v21 )
        {
          if ( &_pthread_key_create )
          {
            v22 = _InterlockedExchangeAdd(v21 + 2, 0xFFFFFFFF);
          }
          else
          {
            v22 = *((_DWORD *)v21 + 2);
            *((_DWORD *)v21 + 2) = v22 - 1;
          }
          if ( v22 == 1 )
          {
            v19 = *(void (**)())(*(_QWORD *)v21 + 16LL);
            if ( v19 != nullsub_25 )
            {
              v58 = v8;
              ((void (__fastcall *)(volatile signed __int32 *))v19)(v21);
              v8 = v58;
            }
            if ( &_pthread_key_create )
            {
              v20 = _InterlockedExchangeAdd(v21 + 3, 0xFFFFFFFF);
            }
            else
            {
              v20 = *((_DWORD *)v21 + 3);
              *((_DWORD *)v21 + 3) = v20 - 1;
            }
            if ( v20 == 1 )
            {
              v56 = v8;
              v46 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v21 + 24LL);
              if ( v46 == sub_9C26E0 )
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v21 + 8LL))(v21);
              else
                v46((__int64)v21);
              v8 = v56;
            }
          }
        }
        v7 += 16;
      }
    }
    v38 = (char *)(*(_QWORD *)a1 + v8);
LABEL_70:
    *(_QWORD *)(a1 + 8) = v38;
  }
}
