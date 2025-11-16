// Function: sub_9D06B0
// Address: 0x9d06b0
//
void __fastcall sub_9D06B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // r15
  __int64 v15; // r12
  volatile signed __int32 *v16; // r13
  signed __int32 v17; // eax
  void (*v18)(); // rax
  signed __int32 v19; // eax
  __int64 (__fastcall *v20)(__int64); // rsi
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rdi
  __int64 v25; // r13
  __int64 v26; // r14
  volatile signed __int32 *v27; // r15
  signed __int32 v28; // eax
  void (*v29)(); // rax
  signed __int32 v30; // eax
  __int64 (__fastcall *v31)(__int64); // rdx
  __int64 v32; // r9
  __int64 v33; // r9
  __int64 v34; // rbx
  __int64 v35; // rdi
  __int64 v36; // r15
  __int64 v37; // r12
  volatile signed __int32 *v38; // r13
  signed __int32 v39; // eax
  void (*v40)(); // rax
  signed __int32 v41; // eax
  __int64 (__fastcall *v42)(__int64); // rcx
  __int64 v43; // rbx
  __int64 v44; // rdi
  __int64 v45; // r15
  __int64 v46; // r12
  volatile signed __int32 *v47; // r13
  signed __int32 v48; // eax
  void (*v49)(); // rax
  signed __int32 v50; // eax
  __int64 (__fastcall *v51)(__int64); // rsi
  __int64 v52; // rbx
  __int64 v53; // r13
  __int64 v54; // r14
  __int64 v55; // r12
  __int64 v56; // r13
  __int64 v57; // rbx
  volatile signed __int32 *v58; // r14
  signed __int32 v59; // eax
  void (*v60)(); // rax
  signed __int32 v61; // eax
  __int64 (__fastcall *v62)(__int64); // rdx
  __int64 v63; // rax
  __int64 v64; // rbx
  __int64 v65; // r13
  __int64 v66; // rdi
  __int64 v67; // r14
  __int64 v68; // r12
  volatile signed __int32 *v69; // r15
  signed __int32 v70; // eax
  void (*v71)(); // rax
  signed __int32 v72; // eax
  __int64 (__fastcall *v73)(__int64); // rdx
  __int64 v74; // rbx
  __int64 v75; // r13
  __int64 v76; // r9
  __int64 v77; // r15
  __int64 v78; // r10
  __int64 v79; // r12
  volatile signed __int32 *v80; // r14
  signed __int32 v81; // eax
  void (*v82)(); // rax
  signed __int32 v83; // eax
  __int64 (__fastcall *v84)(__int64); // rdx
  __int64 v85; // [rsp+8h] [rbp-68h]
  __int64 v86; // [rsp+18h] [rbp-58h]
  __int64 v87; // [rsp+18h] [rbp-58h]
  __int64 v88; // [rsp+18h] [rbp-58h]
  __int64 v89; // [rsp+20h] [rbp-50h]
  __int64 v90; // [rsp+20h] [rbp-50h]
  __int64 v91; // [rsp+20h] [rbp-50h]
  __int64 v92; // [rsp+20h] [rbp-50h]
  __int64 v93; // [rsp+20h] [rbp-50h]
  unsigned int v94; // [rsp+28h] [rbp-48h]
  __int64 v95; // [rsp+28h] [rbp-48h]
  __int64 v96; // [rsp+30h] [rbp-40h]
  __int64 v97; // [rsp+30h] [rbp-40h]
  __int64 v98; // [rsp+30h] [rbp-40h]
  __int64 v99; // [rsp+30h] [rbp-40h]
  __int64 v101; // [rsp+38h] [rbp-38h]
  __int64 v102; // [rsp+38h] [rbp-38h]

  if ( a1 == a2 )
    return;
  v2 = *(_QWORD *)a1;
  v3 = *(unsigned int *)(a1 + 8);
  v4 = a2;
  v5 = a2 + 16;
  v6 = *(_QWORD *)a1;
  if ( *(_QWORD *)a2 == a2 + 16 )
  {
    v94 = *(_DWORD *)(a2 + 8);
    v7 = v94;
    if ( v94 > v3 )
    {
      if ( v94 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v8 = a2 + 16;
        if ( !v3 )
          goto LABEL_6;
        v52 = a2 + 16;
        v85 = 32 * v3;
        v86 = v2 + 32 * v3;
        while ( 1 )
        {
          v53 = *(_QWORD *)(v2 + 8);
          v54 = *(_QWORD *)(v2 + 16);
          *(_DWORD *)v2 = *(_DWORD *)v52;
          v55 = v53;
          v91 = *(_QWORD *)(v2 + 24);
          *(_QWORD *)(v2 + 8) = *(_QWORD *)(v52 + 8);
          *(_QWORD *)(v2 + 16) = *(_QWORD *)(v52 + 16);
          *(_QWORD *)(v2 + 24) = *(_QWORD *)(v52 + 24);
          *(_QWORD *)(v52 + 8) = 0;
          *(_QWORD *)(v52 + 16) = 0;
          *(_QWORD *)(v52 + 24) = 0;
          if ( v53 == v54 )
            goto LABEL_122;
          v98 = v53;
          v56 = v52;
          v57 = v54;
          do
          {
            while ( 1 )
            {
              v58 = *(volatile signed __int32 **)(v55 + 8);
              if ( !v58 )
                goto LABEL_109;
              if ( &_pthread_key_create )
              {
                v59 = _InterlockedExchangeAdd(v58 + 2, 0xFFFFFFFF);
              }
              else
              {
                v59 = *((_DWORD *)v58 + 2);
                *((_DWORD *)v58 + 2) = v59 - 1;
              }
              if ( v59 != 1 )
                goto LABEL_109;
              v60 = *(void (**)())(*(_QWORD *)v58 + 16LL);
              if ( v60 != nullsub_25 )
                ((void (__fastcall *)(volatile signed __int32 *))v60)(v58);
              if ( &_pthread_key_create )
              {
                v61 = _InterlockedExchangeAdd(v58 + 3, 0xFFFFFFFF);
              }
              else
              {
                v61 = *((_DWORD *)v58 + 3);
                *((_DWORD *)v58 + 3) = v61 - 1;
              }
              if ( v61 != 1 )
                goto LABEL_109;
              v62 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v58 + 24LL);
              if ( v62 == sub_9C26E0 )
                break;
              v62((__int64)v58);
LABEL_109:
              v55 += 16;
              if ( v57 == v55 )
                goto LABEL_121;
            }
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v58 + 8LL))(v58);
            v55 += 16;
          }
          while ( v57 != v55 );
LABEL_121:
          v52 = v56;
          v53 = v98;
LABEL_122:
          if ( v53 )
            j_j___libc_free_0(v53, v91 - v53);
          v52 += 32;
          v2 += 32;
          if ( v2 == v86 )
          {
            v4 = a2;
            v3 = v85;
            v5 = *(_QWORD *)a2;
            v7 = *(unsigned int *)(a2 + 8);
            v6 = *(_QWORD *)a1;
            v8 = *(_QWORD *)a2 + v85;
            goto LABEL_6;
          }
        }
      }
      v63 = 32 * v3;
      if ( v2 + v63 == v2 )
      {
LABEL_155:
        *(_DWORD *)(a1 + 8) = 0;
        sub_9D04A0(a1, v7);
        v5 = *(_QWORD *)v4;
        v7 = *(unsigned int *)(v4 + 8);
        v3 = 0;
        v6 = *(_QWORD *)a1;
        v8 = *(_QWORD *)v4;
LABEL_6:
        v9 = v6 + v3;
        v10 = v5 + 32 * v7;
        v11 = v9 + v10 - v8;
        if ( v10 != v8 )
        {
          do
          {
            if ( v9 )
            {
              *(_DWORD *)v9 = *(_DWORD *)v8;
              *(_QWORD *)(v9 + 8) = *(_QWORD *)(v8 + 8);
              *(_QWORD *)(v9 + 16) = *(_QWORD *)(v8 + 16);
              *(_QWORD *)(v9 + 24) = *(_QWORD *)(v8 + 24);
              *(_QWORD *)(v8 + 24) = 0;
              *(_QWORD *)(v8 + 16) = 0;
              *(_QWORD *)(v8 + 8) = 0;
            }
            v9 += 32;
            v8 += 32;
          }
          while ( v9 != v11 );
        }
        *(_DWORD *)(a1 + 8) = v94;
        v101 = *(_QWORD *)v4;
        if ( *(_QWORD *)v4 != *(_QWORD *)v4 + 32LL * *(unsigned int *)(v4 + 8) )
        {
          v96 = v4;
          v12 = *(_QWORD *)v4 + 32LL * *(unsigned int *)(v4 + 8);
          while ( 1 )
          {
            v13 = *(_QWORD *)(v12 - 24);
            v14 = *(_QWORD *)(v12 - 16);
            v12 -= 32;
            v15 = v13;
            if ( v14 != v13 )
              break;
LABEL_27:
            if ( v13 )
              j_j___libc_free_0(v13, *(_QWORD *)(v12 + 24) - v13);
            if ( v101 == v12 )
              goto LABEL_30;
          }
          while ( 1 )
          {
            v16 = *(volatile signed __int32 **)(v15 + 8);
            if ( !v16 )
              goto LABEL_14;
            if ( &_pthread_key_create )
            {
              v17 = _InterlockedExchangeAdd(v16 + 2, 0xFFFFFFFF);
            }
            else
            {
              v17 = *((_DWORD *)v16 + 2);
              *((_DWORD *)v16 + 2) = v17 - 1;
            }
            if ( v17 != 1 )
              goto LABEL_14;
            v18 = *(void (**)())(*(_QWORD *)v16 + 16LL);
            if ( v18 != nullsub_25 )
              ((void (__fastcall *)(volatile signed __int32 *))v18)(v16);
            if ( &_pthread_key_create )
            {
              v19 = _InterlockedExchangeAdd(v16 + 3, 0xFFFFFFFF);
            }
            else
            {
              v19 = *((_DWORD *)v16 + 3);
              *((_DWORD *)v16 + 3) = v19 - 1;
            }
            if ( v19 != 1 )
              goto LABEL_14;
            v20 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v16 + 24LL);
            if ( v20 == sub_9C26E0 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 8LL))(v16);
              v15 += 16;
              if ( v14 == v15 )
              {
LABEL_26:
                v13 = *(_QWORD *)(v12 + 8);
                goto LABEL_27;
              }
            }
            else
            {
              v20((__int64)v16);
LABEL_14:
              v15 += 16;
              if ( v14 == v15 )
                goto LABEL_26;
            }
          }
        }
        goto LABEL_31;
      }
      v64 = v2 + v63;
      v65 = v2;
      while ( 1 )
      {
        v66 = *(_QWORD *)(v64 - 24);
        v67 = *(_QWORD *)(v64 - 16);
        v64 -= 32;
        v68 = v66;
        if ( v67 != v66 )
          break;
LABEL_151:
        if ( v66 )
          j_j___libc_free_0(v66, *(_QWORD *)(v64 + 24) - v66);
        if ( v64 == v65 )
        {
          v7 = v94;
          v4 = a2;
          goto LABEL_155;
        }
      }
      while ( 1 )
      {
        v69 = *(volatile signed __int32 **)(v68 + 8);
        if ( !v69 )
          goto LABEL_138;
        if ( &_pthread_key_create )
        {
          v70 = _InterlockedExchangeAdd(v69 + 2, 0xFFFFFFFF);
        }
        else
        {
          v70 = *((_DWORD *)v69 + 2);
          *((_DWORD *)v69 + 2) = v70 - 1;
        }
        if ( v70 != 1 )
          goto LABEL_138;
        v71 = *(void (**)())(*(_QWORD *)v69 + 16LL);
        if ( v71 != nullsub_25 )
          ((void (__fastcall *)(volatile signed __int32 *))v71)(v69);
        if ( &_pthread_key_create )
        {
          v72 = _InterlockedExchangeAdd(v69 + 3, 0xFFFFFFFF);
        }
        else
        {
          v72 = *((_DWORD *)v69 + 3);
          *((_DWORD *)v69 + 3) = v72 - 1;
        }
        if ( v72 != 1 )
          goto LABEL_138;
        v73 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v69 + 24LL);
        if ( v73 == sub_9C26E0 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v69 + 8LL))(v69);
          v68 += 16;
          if ( v67 == v68 )
          {
LABEL_150:
            v66 = *(_QWORD *)(v64 + 8);
            goto LABEL_151;
          }
        }
        else
        {
          v73((__int64)v69);
LABEL_138:
          v68 += 16;
          if ( v67 == v68 )
            goto LABEL_150;
        }
      }
    }
    v32 = v2;
    if ( !v94 )
    {
LABEL_64:
      v33 = 32 * v3 + v32;
      if ( v33 == v2 )
        goto LABEL_85;
      v97 = v2;
      v90 = v4;
      v34 = v33;
      while ( 1 )
      {
        v35 = *(_QWORD *)(v34 - 24);
        v36 = *(_QWORD *)(v34 - 16);
        v34 -= 32;
        v37 = v35;
        if ( v36 != v35 )
          break;
LABEL_81:
        if ( v35 )
          j_j___libc_free_0(v35, *(_QWORD *)(v34 + 24) - v35);
        if ( v97 == v34 )
        {
          v4 = v90;
LABEL_85:
          *(_DWORD *)(a1 + 8) = v94;
          v102 = *(_QWORD *)v4;
          if ( *(_QWORD *)v4 == *(_QWORD *)v4 + 32LL * *(unsigned int *)(v4 + 8) )
            goto LABEL_31;
          v96 = v4;
          v43 = *(_QWORD *)v4 + 32LL * *(unsigned int *)(v4 + 8);
          while ( 1 )
          {
            v44 = *(_QWORD *)(v43 - 24);
            v45 = *(_QWORD *)(v43 - 16);
            v43 -= 32;
            v46 = v44;
            if ( v45 != v44 )
              break;
LABEL_102:
            if ( v44 )
              j_j___libc_free_0(v44, *(_QWORD *)(v43 + 24) - v44);
            if ( v102 == v43 )
            {
LABEL_30:
              v4 = v96;
LABEL_31:
              *(_DWORD *)(v4 + 8) = 0;
              return;
            }
          }
          while ( 1 )
          {
            v47 = *(volatile signed __int32 **)(v46 + 8);
            if ( !v47 )
              goto LABEL_89;
            if ( &_pthread_key_create )
            {
              v48 = _InterlockedExchangeAdd(v47 + 2, 0xFFFFFFFF);
            }
            else
            {
              v48 = *((_DWORD *)v47 + 2);
              *((_DWORD *)v47 + 2) = v48 - 1;
            }
            if ( v48 != 1 )
              goto LABEL_89;
            v49 = *(void (**)())(*(_QWORD *)v47 + 16LL);
            if ( v49 != nullsub_25 )
              ((void (__fastcall *)(volatile signed __int32 *))v49)(v47);
            if ( &_pthread_key_create )
            {
              v50 = _InterlockedExchangeAdd(v47 + 3, 0xFFFFFFFF);
            }
            else
            {
              v50 = *((_DWORD *)v47 + 3);
              *((_DWORD *)v47 + 3) = v50 - 1;
            }
            if ( v50 != 1 )
              goto LABEL_89;
            v51 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v47 + 24LL);
            if ( v51 == sub_9C26E0 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v47 + 8LL))(v47);
              v46 += 16;
              if ( v45 == v46 )
              {
LABEL_101:
                v44 = *(_QWORD *)(v43 + 8);
                goto LABEL_102;
              }
            }
            else
            {
              v51((__int64)v47);
LABEL_89:
              v46 += 16;
              if ( v45 == v46 )
                goto LABEL_101;
            }
          }
        }
      }
      while ( 1 )
      {
        v38 = *(volatile signed __int32 **)(v37 + 8);
        if ( !v38 )
          goto LABEL_68;
        if ( &_pthread_key_create )
        {
          v39 = _InterlockedExchangeAdd(v38 + 2, 0xFFFFFFFF);
        }
        else
        {
          v39 = *((_DWORD *)v38 + 2);
          *((_DWORD *)v38 + 2) = v39 - 1;
        }
        if ( v39 != 1 )
          goto LABEL_68;
        v40 = *(void (**)())(*(_QWORD *)v38 + 16LL);
        if ( v40 != nullsub_25 )
          ((void (__fastcall *)(volatile signed __int32 *))v40)(v38);
        if ( &_pthread_key_create )
        {
          v41 = _InterlockedExchangeAdd(v38 + 3, 0xFFFFFFFF);
        }
        else
        {
          v41 = *((_DWORD *)v38 + 3);
          *((_DWORD *)v38 + 3) = v41 - 1;
        }
        if ( v41 != 1 )
          goto LABEL_68;
        v42 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v38 + 24LL);
        if ( v42 == sub_9C26E0 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v38 + 8LL))(v38);
          v37 += 16;
          if ( v36 == v37 )
          {
LABEL_80:
            v35 = *(_QWORD *)(v34 + 8);
            goto LABEL_81;
          }
        }
        else
        {
          v42((__int64)v38);
LABEL_68:
          v37 += 16;
          if ( v36 == v37 )
            goto LABEL_80;
        }
      }
    }
    v74 = a2 + 16;
    v75 = v2;
    v99 = v2 + 32LL * v94;
    while ( 1 )
    {
      v76 = *(_QWORD *)(v75 + 8);
      v77 = *(_QWORD *)(v75 + 16);
      v78 = *(_QWORD *)(v75 + 24);
      *(_DWORD *)v75 = *(_DWORD *)v74;
      v79 = v76;
      *(_QWORD *)(v75 + 8) = *(_QWORD *)(v74 + 8);
      *(_QWORD *)(v75 + 16) = *(_QWORD *)(v74 + 16);
      *(_QWORD *)(v75 + 24) = *(_QWORD *)(v74 + 24);
      *(_QWORD *)(v74 + 8) = 0;
      *(_QWORD *)(v74 + 16) = 0;
      *(_QWORD *)(v74 + 24) = 0;
      if ( v76 != v77 )
        break;
LABEL_175:
      if ( v76 )
        j_j___libc_free_0(v76, v78 - v76);
      v74 += 32;
      v75 += 32;
      if ( v75 == v99 )
      {
        v4 = a2;
        v2 = v75;
        v32 = *(_QWORD *)a1;
        v3 = *(unsigned int *)(a1 + 8);
        goto LABEL_64;
      }
    }
    while ( 1 )
    {
      v80 = *(volatile signed __int32 **)(v79 + 8);
      if ( !v80 )
        goto LABEL_163;
      if ( &_pthread_key_create )
      {
        v81 = _InterlockedExchangeAdd(v80 + 2, 0xFFFFFFFF);
      }
      else
      {
        v81 = *((_DWORD *)v80 + 2);
        *((_DWORD *)v80 + 2) = v81 - 1;
      }
      if ( v81 != 1 )
        goto LABEL_163;
      v82 = *(void (**)())(*(_QWORD *)v80 + 16LL);
      if ( v82 != nullsub_25 )
      {
        v88 = v78;
        v93 = v76;
        ((void (__fastcall *)(volatile signed __int32 *))v82)(v80);
        v78 = v88;
        v76 = v93;
      }
      if ( &_pthread_key_create )
      {
        v83 = _InterlockedExchangeAdd(v80 + 3, 0xFFFFFFFF);
      }
      else
      {
        v83 = *((_DWORD *)v80 + 3);
        *((_DWORD *)v80 + 3) = v83 - 1;
      }
      if ( v83 != 1 )
        goto LABEL_163;
      v87 = v78;
      v92 = v76;
      v84 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v80 + 24LL);
      if ( v84 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v80 + 8LL))(v80);
        v79 += 16;
        v76 = v92;
        v78 = v87;
        if ( v77 == v79 )
          goto LABEL_175;
      }
      else
      {
        v84((__int64)v80);
        v78 = v87;
        v76 = v92;
LABEL_163:
        v79 += 16;
        if ( v77 == v79 )
          goto LABEL_175;
      }
    }
  }
  v21 = 32 * v3;
  if ( v2 + v21 != v2 )
  {
    v95 = a2 + 16;
    v22 = v2 + v21;
    v23 = v2;
    v89 = a2;
    while ( 1 )
    {
      v24 = *(_QWORD *)(v22 - 24);
      v25 = *(_QWORD *)(v22 - 16);
      v22 -= 32;
      v26 = v24;
      if ( v25 != v24 )
        break;
LABEL_52:
      if ( v24 )
      {
        a2 = *(_QWORD *)(v22 + 24) - v24;
        j_j___libc_free_0(v24, a2);
      }
      if ( v22 == v23 )
      {
        v5 = v95;
        v4 = v89;
        v2 = *(_QWORD *)a1;
        goto LABEL_56;
      }
    }
    while ( 1 )
    {
      v27 = *(volatile signed __int32 **)(v26 + 8);
      if ( !v27 )
        goto LABEL_39;
      if ( &_pthread_key_create )
      {
        v28 = _InterlockedExchangeAdd(v27 + 2, 0xFFFFFFFF);
      }
      else
      {
        v28 = *((_DWORD *)v27 + 2);
        a2 = (unsigned int)(v28 - 1);
        *((_DWORD *)v27 + 2) = a2;
      }
      if ( v28 != 1 )
        goto LABEL_39;
      v29 = *(void (**)())(*(_QWORD *)v27 + 16LL);
      if ( v29 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v29)(v27);
      if ( &_pthread_key_create )
      {
        v30 = _InterlockedExchangeAdd(v27 + 3, 0xFFFFFFFF);
      }
      else
      {
        v30 = *((_DWORD *)v27 + 3);
        *((_DWORD *)v27 + 3) = v30 - 1;
      }
      if ( v30 != 1 )
        goto LABEL_39;
      v31 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v27 + 24LL);
      if ( v31 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v27 + 8LL))(v27);
        v26 += 16;
        if ( v25 == v26 )
        {
LABEL_51:
          v24 = *(_QWORD *)(v22 + 8);
          goto LABEL_52;
        }
      }
      else
      {
        v31((__int64)v27);
LABEL_39:
        v26 += 16;
        if ( v25 == v26 )
          goto LABEL_51;
      }
    }
  }
LABEL_56:
  if ( v2 != a1 + 16 )
    _libc_free(v2, a2);
  *(_QWORD *)a1 = *(_QWORD *)v4;
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(v4 + 8);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(v4 + 12);
  *(_QWORD *)v4 = v5;
  *(_QWORD *)(v4 + 8) = 0;
}
