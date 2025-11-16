// Function: sub_A05260
// Address: 0xa05260
//
void __fastcall sub_A05260(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // rbx
  unsigned __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 i; // r15
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r14
  _QWORD *v12; // rax
  char *v13; // rsi
  char *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // r14
  __int64 v19; // rbx
  volatile signed __int32 *v20; // r12
  signed __int32 v21; // edx
  void (*v22)(); // rdx
  signed __int32 v23; // eax
  __int64 (__fastcall *v24)(__int64); // rdx
  __int64 v25; // r12
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // r15
  volatile signed __int32 *v29; // rdi
  signed __int32 v30; // edx
  void (*v31)(void); // rdx
  signed __int32 v32; // eax
  void (*v33)(void); // rdx
  __int64 v34; // r12
  __int64 v35; // r15
  __int64 v36; // r14
  __int64 v37; // rdx
  char **v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rbx
  __int64 v41; // r12
  unsigned __int64 v42; // r15
  int v43; // eax
  unsigned int v44; // [rsp-4Ch] [rbp-4Ch]
  __int64 v45; // [rsp-48h] [rbp-48h]
  __int64 v46; // [rsp-40h] [rbp-40h]
  unsigned __int64 v47; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v3 = a2;
    v4 = a1;
    v5 = *(_QWORD *)a1;
    v6 = *(unsigned int *)(a1 + 8);
    v44 = *(_DWORD *)(a2 + 8);
    v45 = v44;
    v46 = *(_QWORD *)a1;
    if ( v44 > v6 )
    {
      if ( v44 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        if ( *(_DWORD *)(a1 + 8) )
        {
          v6 *= 32LL;
          v40 = v5 + 8;
          v41 = *(_QWORD *)a2 + 8LL;
          v42 = v41 + v6;
          do
          {
            v43 = *(_DWORD *)(v41 - 8);
            a2 = v41;
            a1 = v40;
            v41 += 32;
            v47 = v6;
            v40 += 32;
            *(_DWORD *)(v40 - 40) = v43;
            sub_A020B0(a1, (char **)a2, a3, v6);
            v6 = v47;
          }
          while ( v41 != v42 );
          v5 = *(_QWORD *)v4;
          v45 = *(unsigned int *)(v3 + 8);
        }
        goto LABEL_5;
      }
      v25 = v5 + 32 * v6;
LABEL_48:
      if ( v25 == v5 )
      {
LABEL_66:
        *(_DWORD *)(v4 + 8) = 0;
        a2 = v44;
        a1 = v4;
        sub_9D04A0(v4, v44);
        v5 = *(_QWORD *)v4;
        v6 = 0;
        v45 = *(unsigned int *)(v3 + 8);
LABEL_5:
        v7 = v6 + v5;
        v8 = *(_QWORD *)v3 + 32 * v45;
        for ( i = v6 + *(_QWORD *)v3; v8 != i; v7 += 32 )
        {
          if ( v7 )
          {
            *(_DWORD *)v7 = *(_DWORD *)i;
            v10 = *(_QWORD *)(i + 16) - *(_QWORD *)(i + 8);
            *(_QWORD *)(v7 + 8) = 0;
            *(_QWORD *)(v7 + 16) = 0;
            v11 = v10;
            *(_QWORD *)(v7 + 24) = 0;
            if ( v10 )
            {
              if ( v10 > 0x7FFFFFFFFFFFFFF0LL )
                sub_4261EA(a1, a2, v10, v6);
              a1 = v10;
              v12 = (_QWORD *)sub_22077B0(v10);
            }
            else
            {
              v12 = 0;
            }
            *(_QWORD *)(v7 + 8) = v12;
            *(_QWORD *)(v7 + 16) = v12;
            *(_QWORD *)(v7 + 24) = (char *)v12 + v11;
            v13 = *(char **)(i + 16);
            v14 = *(char **)(i + 8);
            if ( v13 == v14 )
            {
              a2 = (__int64)v12;
            }
            else
            {
              a2 = (__int64)v12 + v13 - v14;
              do
              {
                if ( v12 )
                {
                  *v12 = *(_QWORD *)v14;
                  v6 = *((_QWORD *)v14 + 1);
                  v12[1] = v6;
                  if ( v6 )
                  {
                    if ( &_pthread_key_create )
                      _InterlockedAdd((volatile signed __int32 *)(v6 + 8), 1u);
                    else
                      ++*(_DWORD *)(v6 + 8);
                  }
                }
                v12 += 2;
                v14 += 16;
              }
              while ( (_QWORD *)a2 != v12 );
            }
            *(_QWORD *)(v7 + 16) = a2;
          }
          i += 32LL;
        }
LABEL_19:
        *(_DWORD *)(v4 + 8) = v44;
        return;
      }
      while ( 1 )
      {
        v26 = *(_QWORD *)(v25 - 24);
        v27 = *(_QWORD *)(v25 - 16);
        v25 -= 32;
        v28 = v26;
        if ( v27 != v26 )
          break;
LABEL_64:
        if ( !v26 )
          goto LABEL_48;
        j_j___libc_free_0(v26, *(_QWORD *)(v25 + 24) - v26);
        if ( v25 == v5 )
          goto LABEL_66;
      }
      while ( 1 )
      {
        v29 = *(volatile signed __int32 **)(v28 + 8);
        if ( !v29 )
          goto LABEL_51;
        if ( &_pthread_key_create )
        {
          v30 = _InterlockedExchangeAdd(v29 + 2, 0xFFFFFFFF);
        }
        else
        {
          v30 = *((_DWORD *)v29 + 2);
          *((_DWORD *)v29 + 2) = v30 - 1;
        }
        if ( v30 != 1 )
          goto LABEL_51;
        v31 = *(void (**)(void))(*(_QWORD *)v29 + 16LL);
        if ( v31 != nullsub_25 )
          v31();
        if ( &_pthread_key_create )
        {
          v32 = _InterlockedExchangeAdd(v29 + 3, 0xFFFFFFFF);
        }
        else
        {
          v32 = *((_DWORD *)v29 + 3);
          *((_DWORD *)v29 + 3) = v32 - 1;
        }
        if ( v32 != 1 )
          goto LABEL_51;
        v33 = *(void (**)(void))(*(_QWORD *)v29 + 24LL);
        if ( (char *)v33 == (char *)sub_9C26E0 )
        {
          (*(void (**)(void))(*(_QWORD *)v29 + 8LL))();
          v28 += 16;
          if ( v27 == v28 )
          {
LABEL_63:
            v26 = *(_QWORD *)(v25 + 8);
            goto LABEL_64;
          }
        }
        else
        {
          v33();
LABEL_51:
          v28 += 16;
          if ( v27 == v28 )
            goto LABEL_63;
        }
      }
    }
    v15 = *(_QWORD *)a1;
    if ( v44 )
    {
      v34 = v5 + 8;
      v35 = *(_QWORD *)a2 + 8LL;
      v36 = v35 + 32LL * v44;
      do
      {
        v37 = *(unsigned int *)(v35 - 8);
        v38 = (char **)v35;
        v39 = v34;
        v35 += 32;
        v34 += 32;
        *(_DWORD *)(v34 - 40) = v37;
        sub_A020B0(v39, v38, v37, v6);
      }
      while ( v36 != v35 );
      v15 = *(_QWORD *)v4;
      v46 = 32LL * v44 + v5;
      v6 = *(unsigned int *)(v4 + 8);
    }
    v16 = v15 + 32 * v6;
    if ( v16 == v46 )
      goto LABEL_19;
    while ( 1 )
    {
      v17 = *(_QWORD *)(v16 - 24);
      v18 = *(_QWORD *)(v16 - 16);
      v16 -= 32;
      v19 = v17;
      if ( v18 != v17 )
        break;
LABEL_39:
      if ( v17 )
        j_j___libc_free_0(v17, *(_QWORD *)(v16 + 24) - v17);
      if ( v46 == v16 )
        goto LABEL_19;
    }
    while ( 1 )
    {
      v20 = *(volatile signed __int32 **)(v19 + 8);
      if ( !v20 )
        goto LABEL_26;
      if ( &_pthread_key_create )
      {
        v21 = _InterlockedExchangeAdd(v20 + 2, 0xFFFFFFFF);
      }
      else
      {
        v21 = *((_DWORD *)v20 + 2);
        *((_DWORD *)v20 + 2) = v21 - 1;
      }
      if ( v21 != 1 )
        goto LABEL_26;
      v22 = *(void (**)())(*(_QWORD *)v20 + 16LL);
      if ( v22 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v22)(v20);
      if ( &_pthread_key_create )
      {
        v23 = _InterlockedExchangeAdd(v20 + 3, 0xFFFFFFFF);
      }
      else
      {
        v23 = *((_DWORD *)v20 + 3);
        *((_DWORD *)v20 + 3) = v23 - 1;
      }
      if ( v23 != 1 )
        goto LABEL_26;
      v24 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v20 + 24LL);
      if ( v24 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v20 + 8LL))(v20);
        v19 += 16;
        if ( v18 == v19 )
        {
LABEL_38:
          v17 = *(_QWORD *)(v16 + 8);
          goto LABEL_39;
        }
      }
      else
      {
        v24((__int64)v20);
LABEL_26:
        v19 += 16;
        if ( v18 == v19 )
          goto LABEL_38;
      }
    }
  }
}
