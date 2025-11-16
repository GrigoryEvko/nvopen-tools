// Function: sub_3550AE0
// Address: 0x3550ae0
//
void __fastcall sub_3550AE0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int64 v10; // r14
  volatile signed __int32 *v11; // r13
  signed __int32 v12; // edx
  volatile signed __int32 *v13; // r13
  signed __int32 v14; // edx
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 *v17; // rdx
  unsigned __int64 *v18; // rax
  unsigned __int64 *v19; // rcx
  unsigned __int64 *v20; // r14
  unsigned __int64 *v21; // r13
  __int64 (*v22)(); // rax
  __int64 v23; // rdi
  __int64 (*v24)(); // rcx
  __int64 v25; // rax
  unsigned __int64 v26; // r8
  volatile signed __int32 *v27; // rdi
  signed __int32 v28; // ecx
  volatile signed __int32 *v29; // rdi
  signed __int32 v30; // ecx
  unsigned __int64 *v31; // r15
  unsigned __int64 *v32; // r14
  __int64 v33; // rax
  char **v34; // r8
  unsigned __int64 v35; // rdx
  int v36; // r15d
  char *v37; // rax
  char *v38; // rcx
  char *v39; // rdx
  char **v40; // r10
  unsigned __int64 v41; // rcx
  unsigned __int64 *v42; // rsi
  unsigned __int64 *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  _DWORD *v46; // rax
  _DWORD *i; // rdx
  signed __int32 v48; // eax
  signed __int32 v49; // eax
  signed __int32 v50; // eax
  signed __int32 v51; // eax
  void *v52; // rdi
  unsigned int v53; // r13d
  size_t v54; // rdx
  __int64 v55; // rdi
  char *v56; // r13
  char **v57; // [rsp+8h] [rbp-128h]
  char **v58; // [rsp+8h] [rbp-128h]
  __int64 v59; // [rsp+10h] [rbp-120h]
  __int64 v60; // [rsp+10h] [rbp-120h]
  unsigned __int64 v61; // [rsp+18h] [rbp-118h]
  unsigned __int64 v62; // [rsp+18h] [rbp-118h]
  char **v63; // [rsp+18h] [rbp-118h]
  void **v64; // [rsp+18h] [rbp-118h]
  unsigned __int64 v65; // [rsp+18h] [rbp-118h]
  unsigned __int64 *v66; // [rsp+20h] [rbp-110h]
  __int64 v67; // [rsp+20h] [rbp-110h]
  char *v69; // [rsp+30h] [rbp-100h] BYREF
  __int64 v70; // [rsp+38h] [rbp-F8h]
  _QWORD v71[6]; // [rsp+40h] [rbp-F0h] BYREF
  _BYTE *v72; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v73; // [rsp+78h] [rbp-B8h]
  _BYTE v74[176]; // [rsp+80h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a1 + 48);
  v8 = *(unsigned int *)(a1 + 56);
  *(_DWORD *)(a1 + 480) = a2;
  v9 = v7 + 8 * v8;
  while ( v7 != v9 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v9 - 8);
      v9 -= 8;
      if ( !v10 )
        break;
      v11 = *(volatile signed __int32 **)(v10 + 32);
      if ( v11 )
      {
        if ( &_pthread_key_create )
        {
          v12 = _InterlockedExchangeAdd(v11 + 2, 0xFFFFFFFF);
        }
        else
        {
          v12 = *((_DWORD *)v11 + 2);
          *((_DWORD *)v11 + 2) = v12 - 1;
        }
        if ( v12 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 16LL))(v11);
          if ( &_pthread_key_create )
          {
            v50 = _InterlockedExchangeAdd(v11 + 3, 0xFFFFFFFF);
          }
          else
          {
            v50 = *((_DWORD *)v11 + 3);
            *((_DWORD *)v11 + 3) = v50 - 1;
          }
          if ( v50 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 24LL))(v11);
        }
      }
      v13 = *(volatile signed __int32 **)(v10 + 16);
      if ( v13 )
      {
        if ( &_pthread_key_create )
        {
          v14 = _InterlockedExchangeAdd(v13 + 2, 0xFFFFFFFF);
        }
        else
        {
          v14 = *((_DWORD *)v13 + 2);
          *((_DWORD *)v13 + 2) = v14 - 1;
        }
        if ( v14 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v13 + 16LL))(v13);
          if ( &_pthread_key_create )
          {
            v51 = _InterlockedExchangeAdd(v13 + 3, 0xFFFFFFFF);
          }
          else
          {
            v51 = *((_DWORD *)v13 + 3);
            *((_DWORD *)v13 + 3) = v51 - 1;
          }
          if ( v51 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v13 + 24LL))(v13);
        }
      }
      j_j___libc_free_0(v10);
      if ( v7 == v9 )
        goto LABEL_13;
    }
  }
LABEL_13:
  *(_DWORD *)(a1 + 56) = 0;
  if ( a2 )
  {
    v15 = *(unsigned int *)(a1 + 60);
    v16 = 0;
    if ( a2 > v15 )
    {
      sub_354B1B0(a1 + 48, a2, v15, a4, a5, a6);
      v16 = *(unsigned int *)(a1 + 56);
    }
    v17 = *(unsigned __int64 **)(a1 + 48);
    v18 = &v17[v16];
    v19 = &v17[a2];
    if ( v18 != v19 )
    {
      do
      {
        if ( v18 )
          *v18 = 0;
        ++v18;
      }
      while ( v19 != v18 );
      v17 = *(unsigned __int64 **)(a1 + 48);
    }
    v20 = &v17[a2];
    *(_DWORD *)(a1 + 56) = a2;
    if ( v20 != v17 )
    {
      v21 = v17;
      do
      {
        v22 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 128LL);
        if ( v22 == sub_2DAC790 )
          BUG();
        v23 = v22();
        v24 = *(__int64 (**)())(*(_QWORD *)v23 + 1256LL);
        v25 = 0;
        if ( v24 != sub_2FDC7B0 )
          v25 = ((__int64 (__fastcall *)(__int64, _QWORD))v24)(v23, *(_QWORD *)(a1 + 16));
        v26 = *v21;
        *v21 = v25;
        if ( v26 )
        {
          v27 = *(volatile signed __int32 **)(v26 + 32);
          if ( v27 )
          {
            if ( &_pthread_key_create )
            {
              v28 = _InterlockedExchangeAdd(v27 + 2, 0xFFFFFFFF);
            }
            else
            {
              v28 = *((_DWORD *)v27 + 2);
              *((_DWORD *)v27 + 2) = v28 - 1;
            }
            if ( v28 == 1 )
            {
              v62 = v26;
              (*(void (**)(void))(*(_QWORD *)v27 + 16LL))();
              v26 = v62;
              if ( &_pthread_key_create )
              {
                v49 = _InterlockedExchangeAdd(v27 + 3, 0xFFFFFFFF);
              }
              else
              {
                v49 = *((_DWORD *)v27 + 3);
                *((_DWORD *)v27 + 3) = v49 - 1;
              }
              if ( v49 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v27 + 24LL))(v27);
                v26 = v62;
              }
            }
          }
          v29 = *(volatile signed __int32 **)(v26 + 16);
          if ( v29 )
          {
            if ( &_pthread_key_create )
            {
              v30 = _InterlockedExchangeAdd(v29 + 2, 0xFFFFFFFF);
            }
            else
            {
              v30 = *((_DWORD *)v29 + 2);
              *((_DWORD *)v29 + 2) = v30 - 1;
            }
            if ( v30 == 1 )
            {
              v61 = v26;
              (*(void (**)(void))(*(_QWORD *)v29 + 16LL))();
              v26 = v61;
              if ( &_pthread_key_create )
              {
                v48 = _InterlockedExchangeAdd(v29 + 3, 0xFFFFFFFF);
              }
              else
              {
                v48 = *((_DWORD *)v29 + 3);
                *((_DWORD *)v29 + 3) = v48 - 1;
              }
              if ( v48 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v29 + 24LL))(v29);
                v26 = v61;
              }
            }
          }
          j_j___libc_free_0(v26);
        }
        ++v21;
      }
      while ( v20 != v21 );
    }
  }
  v31 = *(unsigned __int64 **)(a1 + 112);
  v32 = &v31[18 * *(unsigned int *)(a1 + 120)];
  while ( v31 != v32 )
  {
    v32 -= 18;
    if ( (unsigned __int64 *)*v32 != v32 + 2 )
      _libc_free(*v32);
  }
  v33 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 120) = 0;
  v34 = &v69;
  v69 = (char *)v71;
  v35 = *(unsigned int *)(v33 + 48);
  v70 = 0x600000000LL;
  v36 = v35;
  if ( !v35 )
  {
    v72 = v74;
    v73 = 0x1000000000LL;
    if ( !a2 )
      goto LABEL_56;
    v41 = *(_QWORD *)(a1 + 112);
    v40 = &v72;
    goto LABEL_89;
  }
  v37 = (char *)v71;
  v38 = (char *)v71;
  if ( v35 > 6 )
  {
    v65 = v35;
    sub_C8D5F0((__int64)&v69, v71, v35, 8u, (__int64)&v69, a6);
    v38 = v69;
    v37 = &v69[8 * (unsigned int)v70];
    v39 = &v69[8 * v65];
    if ( v39 == v37 )
      goto LABEL_47;
  }
  else
  {
    v39 = (char *)&v71[v35];
    if ( v39 == (char *)v71 )
      goto LABEL_47;
  }
  do
  {
    if ( v37 )
      *(_QWORD *)v37 = 0;
    v37 += 8;
  }
  while ( v39 != v37 );
LABEL_47:
  LODWORD(v70) = v36;
  v72 = v74;
  v73 = 0x1000000000LL;
  sub_353DFE0((__int64)&v72, &v69, (__int64)v39, (__int64)v38, (__int64)&v69, a6);
  v35 = *(unsigned int *)(a1 + 120);
  v40 = &v72;
  if ( a2 != v35 )
  {
    v41 = *(_QWORD *)(a1 + 112);
    if ( a2 < v35 )
    {
      v42 = (unsigned __int64 *)(v41 + 144LL * a2);
      v43 = (unsigned __int64 *)(v41 + 144 * v35);
      while ( v42 != v43 )
      {
        v43 -= 18;
        if ( (unsigned __int64 *)*v43 != v43 + 2 )
        {
          v66 = v43;
          _libc_free(*v43);
          v43 = v66;
        }
      }
      *(_DWORD *)(a1 + 120) = a2;
      goto LABEL_54;
    }
LABEL_89:
    v67 = a2 - v35;
    if ( a2 > (unsigned __int64)*(unsigned int *)(a1 + 124) )
    {
      v55 = a1 + 112;
      if ( v41 > (unsigned __int64)&v72 || (unsigned __int64)&v72 >= v41 + 144 * v35 )
      {
        sub_35509D0(v55, a2, v35, v41, (__int64)v34, a6);
        v41 = *(_QWORD *)(a1 + 112);
        v40 = &v72;
      }
      else
      {
        v56 = (char *)&v72 - v41;
        sub_35509D0(v55, a2, v35, v41, (__int64)v34, a6);
        v41 = *(_QWORD *)(a1 + 112);
        v40 = (char **)&v56[v41];
      }
    }
    a6 = v67;
    v34 = (char **)(v41 + 144LL * *(unsigned int *)(a1 + 120));
    do
    {
      while ( 1 )
      {
        if ( v34 )
        {
          v52 = v34 + 2;
          *((_DWORD *)v34 + 2) = 0;
          *v34 = (char *)(v34 + 2);
          *((_DWORD *)v34 + 3) = 16;
          v53 = *((_DWORD *)v40 + 2);
          if ( v34 != v40 )
          {
            if ( v53 )
              break;
          }
        }
        v34 += 18;
        if ( !--a6 )
          goto LABEL_98;
      }
      v54 = 8LL * v53;
      if ( v53 <= 0x10 )
        goto LABEL_96;
      v58 = v40;
      v60 = a6;
      v64 = (void **)v34;
      sub_C8D5F0((__int64)v34, v34 + 2, v53, 8u, (__int64)v34, a6);
      v40 = v58;
      v34 = (char **)v64;
      a6 = v60;
      v52 = *v64;
      v54 = 8LL * *((unsigned int *)v58 + 2);
      if ( v54 )
      {
LABEL_96:
        v57 = v34;
        v59 = a6;
        v63 = v40;
        memcpy(v52, *v40, v54);
        v34 = v57;
        a6 = v59;
        v40 = v63;
      }
      *((_DWORD *)v34 + 2) = v53;
      v34 += 18;
      --a6;
    }
    while ( a6 );
LABEL_98:
    *(_DWORD *)(a1 + 120) += v67;
  }
LABEL_54:
  if ( v72 != v74 )
    _libc_free((unsigned __int64)v72);
LABEL_56:
  if ( v69 != (char *)v71 )
    _libc_free((unsigned __int64)v69);
  *(_DWORD *)(a1 + 280) = 0;
  if ( a2 )
  {
    v44 = 0;
    if ( a2 > (unsigned __int64)*(unsigned int *)(a1 + 284) )
    {
      sub_C8D5F0(a1 + 272, (const void *)(a1 + 288), a2, 4u, (__int64)v34, a6);
      v44 = 4LL * *(unsigned int *)(a1 + 280);
    }
    v45 = *(_QWORD *)(a1 + 272);
    v46 = (_DWORD *)(v45 + v44);
    for ( i = (_DWORD *)(v45 + 4LL * a2); i != v46; ++v46 )
    {
      if ( v46 )
        *v46 = 0;
    }
    *(_DWORD *)(a1 + 280) = a2;
  }
}
