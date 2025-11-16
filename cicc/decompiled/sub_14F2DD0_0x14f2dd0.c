// Function: sub_14F2DD0
// Address: 0x14f2dd0
//
void __fastcall sub_14F2DD0(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rcx
  __int64 *v4; // r12
  __int64 v5; // rcx
  unsigned __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rbx
  volatile signed __int32 *v10; // r13
  signed __int32 v11; // eax
  signed __int32 v12; // eax
  unsigned __int64 v13; // rbx
  __int64 v14; // rcx
  unsigned __int64 v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // r13
  __int64 v18; // r14
  volatile signed __int32 *v19; // r12
  signed __int32 v20; // eax
  signed __int32 v21; // eax
  __int64 v22; // rax
  __int64 *v23; // r12
  unsigned __int64 v24; // r14
  unsigned __int64 v25; // rcx
  __int64 v26; // rbx
  __int64 v27; // rdi
  __int64 v28; // r14
  __int64 v29; // r13
  volatile signed __int32 *v30; // r12
  signed __int32 v31; // eax
  signed __int32 v32; // eax
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rcx
  unsigned __int64 v35; // r13
  __int64 v36; // rdi
  __int64 v37; // r14
  __int64 v38; // rbx
  volatile signed __int32 *v39; // r12
  signed __int32 v40; // eax
  signed __int32 v41; // eax
  __int64 v42; // rbx
  __int64 v43; // rdi
  __int64 v44; // r15
  __int64 v45; // r13
  volatile signed __int32 *v46; // r12
  signed __int32 v47; // eax
  signed __int32 v48; // eax
  unsigned __int64 v49; // rbx
  __int64 v50; // r9
  __int64 v51; // r13
  __int64 v52; // r14
  volatile signed __int32 *v53; // r15
  signed __int32 v54; // eax
  signed __int32 v55; // eax
  __int64 *v56; // r13
  unsigned __int64 v57; // r12
  __int64 v58; // r15
  __int64 v59; // rcx
  __int64 v60; // rbx
  __int64 v61; // r12
  volatile signed __int32 *v62; // r14
  signed __int32 v63; // eax
  signed __int32 v64; // eax
  __int64 v65; // [rsp+0h] [rbp-70h]
  unsigned __int64 v66; // [rsp+8h] [rbp-68h]
  unsigned __int64 v67; // [rsp+8h] [rbp-68h]
  __int64 v68; // [rsp+10h] [rbp-60h]
  __int64 v69; // [rsp+10h] [rbp-60h]
  unsigned int v70; // [rsp+1Ch] [rbp-54h]
  unsigned __int64 v71; // [rsp+20h] [rbp-50h]
  unsigned __int64 v73; // [rsp+38h] [rbp-38h]
  __int64 v74; // [rsp+38h] [rbp-38h]
  unsigned __int64 v75; // [rsp+38h] [rbp-38h]
  __int64 v76; // [rsp+38h] [rbp-38h]
  __int64 v77; // [rsp+38h] [rbp-38h]

  if ( (__int64 *)a1 != a2 )
  {
    v2 = *(_QWORD *)a1;
    v3 = *(unsigned int *)(a1 + 8);
    v4 = a2 + 2;
    v73 = *(_QWORD *)a1;
    if ( (__int64 *)*a2 == a2 + 2 )
    {
      v70 = *((_DWORD *)a2 + 2);
      v13 = v70;
      if ( v70 <= v3 )
      {
        v33 = *(_QWORD *)a1;
        if ( v70 )
        {
          v66 = v2 + 32LL * v70;
          v49 = *(_QWORD *)a1;
          do
          {
            v50 = *(_QWORD *)(v49 + 8);
            v51 = *(_QWORD *)(v49 + 16);
            *(_DWORD *)v49 = *(_DWORD *)v4;
            v52 = v50;
            v68 = *(_QWORD *)(v49 + 24);
            *(_QWORD *)(v49 + 8) = v4[1];
            *(_QWORD *)(v49 + 16) = v4[2];
            *(_QWORD *)(v49 + 24) = v4[3];
            v4[1] = 0;
            v4[2] = 0;
            for ( v4[3] = 0; v51 != v52; v50 = v77 )
            {
              while ( 1 )
              {
                v53 = *(volatile signed __int32 **)(v52 + 8);
                if ( v53 )
                {
                  if ( &_pthread_key_create )
                  {
                    v54 = _InterlockedExchangeAdd(v53 + 2, 0xFFFFFFFF);
                  }
                  else
                  {
                    v54 = *((_DWORD *)v53 + 2);
                    *((_DWORD *)v53 + 2) = v54 - 1;
                  }
                  if ( v54 == 1 )
                  {
                    v77 = v50;
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v53 + 16LL))(v53);
                    v50 = v77;
                    if ( &_pthread_key_create )
                    {
                      v55 = _InterlockedExchangeAdd(v53 + 3, 0xFFFFFFFF);
                    }
                    else
                    {
                      v55 = *((_DWORD *)v53 + 3);
                      *((_DWORD *)v53 + 3) = v55 - 1;
                    }
                    if ( v55 == 1 )
                      break;
                  }
                }
                v52 += 16;
                if ( v51 == v52 )
                  goto LABEL_127;
              }
              v52 += 16;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v53 + 24LL))(v53);
            }
LABEL_127:
            if ( v50 )
              j_j___libc_free_0(v50, v68 - v50);
            v4 += 4;
            v49 += 32LL;
          }
          while ( v66 != v49 );
          v2 = v49;
          v33 = *(_QWORD *)a1;
          v3 = *(unsigned int *)(a1 + 8);
        }
        v34 = v33 + 32 * v3;
        if ( v34 != v2 )
        {
          v75 = v2;
          v35 = v34;
          do
          {
            v36 = *(_QWORD *)(v35 - 24);
            v37 = *(_QWORD *)(v35 - 16);
            v35 -= 32LL;
            v38 = v36;
            if ( v37 != v36 )
            {
              do
              {
                while ( 1 )
                {
                  v39 = *(volatile signed __int32 **)(v38 + 8);
                  if ( v39 )
                  {
                    if ( &_pthread_key_create )
                    {
                      v40 = _InterlockedExchangeAdd(v39 + 2, 0xFFFFFFFF);
                    }
                    else
                    {
                      v40 = *((_DWORD *)v39 + 2);
                      *((_DWORD *)v39 + 2) = v40 - 1;
                    }
                    if ( v40 == 1 )
                    {
                      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v39 + 16LL))(v39);
                      if ( &_pthread_key_create )
                      {
                        v41 = _InterlockedExchangeAdd(v39 + 3, 0xFFFFFFFF);
                      }
                      else
                      {
                        v41 = *((_DWORD *)v39 + 3);
                        *((_DWORD *)v39 + 3) = v41 - 1;
                      }
                      if ( v41 == 1 )
                        break;
                    }
                  }
                  v38 += 16;
                  if ( v37 == v38 )
                    goto LABEL_89;
                }
                v38 += 16;
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v39 + 24LL))(v39);
              }
              while ( v37 != v38 );
LABEL_89:
              v36 = *(_QWORD *)(v35 + 8);
            }
            if ( v36 )
              j_j___libc_free_0(v36, *(_QWORD *)(v35 + 24) - v36);
          }
          while ( v75 != v35 );
        }
        *(_DWORD *)(a1 + 8) = v70;
        if ( *a2 != *a2 + 32LL * *((unsigned int *)a2 + 2) )
        {
          v76 = *a2;
          v42 = *a2 + 32LL * *((unsigned int *)a2 + 2);
          do
          {
            v43 = *(_QWORD *)(v42 - 24);
            v44 = *(_QWORD *)(v42 - 16);
            v42 -= 32;
            v45 = v43;
            if ( v44 != v43 )
            {
              do
              {
                while ( 1 )
                {
                  v46 = *(volatile signed __int32 **)(v45 + 8);
                  if ( v46 )
                  {
                    if ( &_pthread_key_create )
                    {
                      v47 = _InterlockedExchangeAdd(v46 + 2, 0xFFFFFFFF);
                    }
                    else
                    {
                      v47 = *((_DWORD *)v46 + 2);
                      *((_DWORD *)v46 + 2) = v47 - 1;
                    }
                    if ( v47 == 1 )
                    {
                      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v46 + 16LL))(v46);
                      if ( &_pthread_key_create )
                      {
                        v48 = _InterlockedExchangeAdd(v46 + 3, 0xFFFFFFFF);
                      }
                      else
                      {
                        v48 = *((_DWORD *)v46 + 3);
                        *((_DWORD *)v46 + 3) = v48 - 1;
                      }
                      if ( v48 == 1 )
                        break;
                    }
                  }
                  v45 += 16;
                  if ( v44 == v45 )
                    goto LABEL_106;
                }
                v45 += 16;
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v46 + 24LL))(v46);
              }
              while ( v44 != v45 );
LABEL_106:
              v43 = *(_QWORD *)(v42 + 8);
            }
            if ( v43 )
              j_j___libc_free_0(v43, *(_QWORD *)(v42 + 24) - v43);
          }
          while ( v76 != v42 );
        }
      }
      else
      {
        if ( v70 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v22 = (__int64)(a2 + 2);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v65 = 32 * v3;
            v67 = v2 + 32 * v3;
            v56 = a2 + 2;
            v57 = *(_QWORD *)a1;
            do
            {
              v58 = *(_QWORD *)(v57 + 8);
              v59 = *(_QWORD *)(v57 + 16);
              *(_DWORD *)v57 = *(_DWORD *)v56;
              v60 = v58;
              v69 = *(_QWORD *)(v57 + 24);
              *(_QWORD *)(v57 + 8) = v56[1];
              *(_QWORD *)(v57 + 16) = v56[2];
              *(_QWORD *)(v57 + 24) = v56[3];
              v56[1] = 0;
              v56[2] = 0;
              v56[3] = 0;
              if ( v58 != v59 )
              {
                v71 = v57;
                v61 = v59;
                do
                {
                  while ( 1 )
                  {
                    v62 = *(volatile signed __int32 **)(v60 + 8);
                    if ( v62 )
                    {
                      if ( &_pthread_key_create )
                      {
                        v63 = _InterlockedExchangeAdd(v62 + 2, 0xFFFFFFFF);
                      }
                      else
                      {
                        v63 = *((_DWORD *)v62 + 2);
                        *((_DWORD *)v62 + 2) = v63 - 1;
                      }
                      if ( v63 == 1 )
                      {
                        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v62 + 16LL))(v62);
                        if ( &_pthread_key_create )
                        {
                          v64 = _InterlockedExchangeAdd(v62 + 3, 0xFFFFFFFF);
                        }
                        else
                        {
                          v64 = *((_DWORD *)v62 + 3);
                          *((_DWORD *)v62 + 3) = v64 - 1;
                        }
                        if ( v64 == 1 )
                          break;
                      }
                    }
                    v60 += 16;
                    if ( v61 == v60 )
                      goto LABEL_145;
                  }
                  v60 += 16;
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v62 + 24LL))(v62);
                }
                while ( v61 != v60 );
LABEL_145:
                v57 = v71;
              }
              if ( v58 )
                j_j___libc_free_0(v58, v69 - v58);
              v56 += 4;
              v57 += 32LL;
            }
            while ( v57 != v67 );
            v3 = v65;
            v4 = (__int64 *)*a2;
            v13 = *((unsigned int *)a2 + 2);
            v73 = *(_QWORD *)a1;
            v22 = *a2 + v65;
          }
        }
        else
        {
          v14 = 32 * v3;
          if ( v73 + v14 != v73 )
          {
            v15 = v73 + v14;
            do
            {
              v16 = *(_QWORD *)(v15 - 24);
              v17 = *(_QWORD *)(v15 - 16);
              v15 -= 32LL;
              v18 = v16;
              if ( v17 != v16 )
              {
                do
                {
                  while ( 1 )
                  {
                    v19 = *(volatile signed __int32 **)(v18 + 8);
                    if ( v19 )
                    {
                      if ( &_pthread_key_create )
                      {
                        v20 = _InterlockedExchangeAdd(v19 + 2, 0xFFFFFFFF);
                      }
                      else
                      {
                        v20 = *((_DWORD *)v19 + 2);
                        *((_DWORD *)v19 + 2) = v20 - 1;
                      }
                      if ( v20 == 1 )
                      {
                        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 16LL))(v19);
                        if ( &_pthread_key_create )
                        {
                          v21 = _InterlockedExchangeAdd(v19 + 3, 0xFFFFFFFF);
                        }
                        else
                        {
                          v21 = *((_DWORD *)v19 + 3);
                          *((_DWORD *)v19 + 3) = v21 - 1;
                        }
                        if ( v21 == 1 )
                          break;
                      }
                    }
                    v18 += 16;
                    if ( v17 == v18 )
                      goto LABEL_40;
                  }
                  v18 += 16;
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 24LL))(v19);
                }
                while ( v17 != v18 );
LABEL_40:
                v16 = *(_QWORD *)(v15 + 8);
              }
              if ( v16 )
                j_j___libc_free_0(v16, *(_QWORD *)(v15 + 24) - v16);
            }
            while ( v15 != v73 );
            v13 = v70;
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_14F2B60(a1, v13);
          v3 = 0;
          v4 = (__int64 *)*a2;
          v13 = *((unsigned int *)a2 + 2);
          v73 = *(_QWORD *)a1;
          v22 = *a2;
        }
        v23 = &v4[4 * v13];
        v24 = v3 + v73;
        v25 = (unsigned __int64)v23 + v3 + v73 - v22;
        if ( v23 != (__int64 *)v22 )
        {
          do
          {
            if ( v24 )
            {
              *(_DWORD *)v24 = *(_DWORD *)v22;
              *(_QWORD *)(v24 + 8) = *(_QWORD *)(v22 + 8);
              *(_QWORD *)(v24 + 16) = *(_QWORD *)(v22 + 16);
              *(_QWORD *)(v24 + 24) = *(_QWORD *)(v22 + 24);
              *(_QWORD *)(v22 + 24) = 0;
              *(_QWORD *)(v22 + 16) = 0;
              *(_QWORD *)(v22 + 8) = 0;
            }
            v24 += 32LL;
            v22 += 32;
          }
          while ( v24 != v25 );
        }
        *(_DWORD *)(a1 + 8) = v70;
        if ( *a2 != *a2 + 32LL * *((unsigned int *)a2 + 2) )
        {
          v74 = *a2;
          v26 = *a2 + 32LL * *((unsigned int *)a2 + 2);
          do
          {
            v27 = *(_QWORD *)(v26 - 24);
            v28 = *(_QWORD *)(v26 - 16);
            v26 -= 32;
            v29 = v27;
            if ( v28 != v27 )
            {
              do
              {
                while ( 1 )
                {
                  v30 = *(volatile signed __int32 **)(v29 + 8);
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
                      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v30 + 16LL))(v30);
                      if ( &_pthread_key_create )
                      {
                        v32 = _InterlockedExchangeAdd(v30 + 3, 0xFFFFFFFF);
                      }
                      else
                      {
                        v32 = *((_DWORD *)v30 + 3);
                        *((_DWORD *)v30 + 3) = v32 - 1;
                      }
                      if ( v32 == 1 )
                        break;
                    }
                  }
                  v29 += 16;
                  if ( v28 == v29 )
                    goto LABEL_66;
                }
                v29 += 16;
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v30 + 24LL))(v30);
              }
              while ( v28 != v29 );
LABEL_66:
              v27 = *(_QWORD *)(v26 + 8);
            }
            if ( v27 )
              j_j___libc_free_0(v27, *(_QWORD *)(v26 + 24) - v27);
          }
          while ( v74 != v26 );
        }
      }
      *((_DWORD *)a2 + 2) = 0;
    }
    else
    {
      v5 = 32 * v3;
      if ( v2 + v5 != v2 )
      {
        v6 = v2 + v5;
        do
        {
          v7 = *(_QWORD *)(v6 - 24);
          v8 = *(_QWORD *)(v6 - 16);
          v6 -= 32LL;
          v9 = v7;
          if ( v8 != v7 )
          {
            do
            {
              while ( 1 )
              {
                v10 = *(volatile signed __int32 **)(v9 + 8);
                if ( v10 )
                {
                  if ( &_pthread_key_create )
                  {
                    v11 = _InterlockedExchangeAdd(v10 + 2, 0xFFFFFFFF);
                  }
                  else
                  {
                    v11 = *((_DWORD *)v10 + 2);
                    *((_DWORD *)v10 + 2) = v11 - 1;
                  }
                  if ( v11 == 1 )
                  {
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 16LL))(v10);
                    if ( &_pthread_key_create )
                    {
                      v12 = _InterlockedExchangeAdd(v10 + 3, 0xFFFFFFFF);
                    }
                    else
                    {
                      v12 = *((_DWORD *)v10 + 3);
                      *((_DWORD *)v10 + 3) = v12 - 1;
                    }
                    if ( v12 == 1 )
                      break;
                  }
                }
                v9 += 16;
                if ( v8 == v9 )
                  goto LABEL_16;
              }
              v9 += 16;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 24LL))(v10);
            }
            while ( v8 != v9 );
LABEL_16:
            v7 = *(_QWORD *)(v6 + 8);
          }
          if ( v7 )
            j_j___libc_free_0(v7, *(_QWORD *)(v6 + 24) - v7);
        }
        while ( v6 != v73 );
        v4 = a2 + 2;
        v2 = *(_QWORD *)a1;
      }
      if ( v2 != a1 + 16 )
        _libc_free(v2);
      *(_QWORD *)a1 = *a2;
      *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
      *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
      *a2 = (__int64)v4;
      a2[1] = 0;
    }
  }
}
