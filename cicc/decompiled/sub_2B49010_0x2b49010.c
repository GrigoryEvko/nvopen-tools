// Function: sub_2B49010
// Address: 0x2b49010
//
void __fastcall sub_2B49010(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r13
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r8
  __int64 *v13; // rdx
  unsigned __int64 v14; // r14
  __int64 *v15; // r15
  __int64 *v16; // r13
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // r13
  __int64 v24; // r12
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // r15
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // r13
  unsigned __int64 v30; // rdi
  __int64 v31; // r13
  __int64 v32; // r12
  unsigned __int64 v33; // rdi
  __int64 v34; // r14
  __int64 v35; // r10
  __int64 v36; // r13
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  unsigned __int64 v43; // r13
  unsigned __int64 v44; // rdi
  __int64 v45; // r13
  __int64 v46; // r8
  _QWORD *v47; // rax
  __int64 v48; // r14
  __int64 v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  unsigned __int64 v54; // [rsp-60h] [rbp-60h]
  unsigned __int64 v55; // [rsp-50h] [rbp-50h]
  int v56; // [rsp-44h] [rbp-44h]
  __int64 v57; // [rsp-40h] [rbp-40h]
  unsigned __int64 v58; // [rsp-40h] [rbp-40h]
  unsigned __int64 v59; // [rsp-40h] [rbp-40h]
  unsigned __int64 v60; // [rsp-40h] [rbp-40h]
  unsigned __int64 v61; // [rsp-40h] [rbp-40h]
  unsigned __int64 v62; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v6 = a2 + 2;
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = *(_QWORD *)a1;
    if ( (__int64 *)*a2 == a2 + 2 )
    {
      v12 = *((unsigned int *)a2 + 2);
      v56 = *((_DWORD *)a2 + 2);
      if ( v12 <= v10 )
      {
        v28 = *(_QWORD *)a1;
        if ( *((_DWORD *)a2 + 2) )
        {
          v45 = (__int64)(a2 + 3);
          v62 = v9 + 104 * v12;
          do
          {
            v46 = v11 + 8;
            *(_QWORD *)v11 = *(_QWORD *)(v45 - 8);
            if ( (*(_BYTE *)(v11 + 16) & 1) == 0 )
            {
              sub_C7D6A0(*(_QWORD *)(v11 + 24), 16LL * *(unsigned int *)(v11 + 32), 8);
              v46 = v11 + 8;
            }
            *(_DWORD *)(v11 + 16) = 1;
            v47 = (_QWORD *)(v11 + 24);
            v48 = v11 + 56;
            *(_DWORD *)(v11 + 20) = 0;
            do
            {
              if ( v47 )
                *v47 = -4096;
              v47 += 2;
            }
            while ( (_QWORD *)v48 != v47 );
            v11 += 104LL;
            sub_2B48D10(v46, v45);
            v49 = v45 + 48;
            v45 += 104;
            sub_2B0BCF0(v48, v49, v50, v51, v52, v53);
          }
          while ( v62 != v11 );
          v28 = *(_QWORD *)a1;
          v10 = *(unsigned int *)(a1 + 8);
        }
        v29 = v28 + 104 * v10;
        while ( v11 != v29 )
        {
          v29 -= 104LL;
          v30 = *(_QWORD *)(v29 + 56);
          if ( v30 != v29 + 72 )
            _libc_free(v30);
          if ( (*(_BYTE *)(v29 + 16) & 1) == 0 )
            sub_C7D6A0(*(_QWORD *)(v29 + 24), 16LL * *(unsigned int *)(v29 + 32), 8);
        }
        *(_DWORD *)(a1 + 8) = v56;
        v31 = *a2;
        v32 = *a2 + 104LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v32 )
        {
          do
          {
            v32 -= 104;
            v33 = *(_QWORD *)(v32 + 56);
            if ( v33 != v32 + 72 )
              _libc_free(v33);
            if ( (*(_BYTE *)(v32 + 16) & 1) == 0 )
              sub_C7D6A0(*(_QWORD *)(v32 + 24), 16LL * *(unsigned int *)(v32 + 32), 8);
          }
          while ( v31 != v32 );
        }
      }
      else
      {
        if ( v12 > *(unsigned int *)(a1 + 12) )
        {
          v42 = 3 * v10;
          v43 = v9 + 104 * v10;
          while ( v9 != v43 )
          {
            while ( 1 )
            {
              v43 -= 104LL;
              v44 = *(_QWORD *)(v43 + 56);
              if ( v44 != v43 + 72 )
              {
                v60 = v12;
                _libc_free(v44);
                v12 = v60;
              }
              if ( (*(_BYTE *)(v43 + 16) & 1) != 0 )
                break;
              v61 = v12;
              sub_C7D6A0(*(_QWORD *)(v43 + 24), 16LL * *(unsigned int *)(v43 + 32), 8);
              v12 = v61;
              if ( v9 == v43 )
                goto LABEL_61;
            }
          }
LABEL_61:
          *(_DWORD *)(a1 + 8) = 0;
          sub_2B48E80(a1, v12, v42, a4, v12, a6);
          v6 = (__int64 *)*a2;
          v12 = *((unsigned int *)a2 + 2);
          v10 = 0;
          v9 = *(_QWORD *)a1;
          v13 = (__int64 *)*a2;
        }
        else
        {
          v13 = a2 + 2;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v34 = (__int64)(a2 + 3);
            v10 *= 104LL;
            v54 = v10;
            v55 = v11 + v10;
            do
            {
              v35 = v11 + 8;
              *(_QWORD *)v11 = *(_QWORD *)(v34 - 8);
              if ( (*(_BYTE *)(v11 + 16) & 1) == 0 )
              {
                v58 = v10;
                sub_C7D6A0(*(_QWORD *)(v11 + 24), 16LL * *(unsigned int *)(v11 + 32), 8);
                v35 = v11 + 8;
                v10 = v58;
              }
              *(_DWORD *)(v11 + 16) = 1;
              v36 = v11 + 56;
              *(_DWORD *)(v11 + 20) = 0;
              if ( v11 == -24 || (*(_QWORD *)(v11 + 24) = -4096, v11 != -40) )
                *(_QWORD *)(v11 + 40) = -4096;
              v59 = v10;
              v11 += 104LL;
              sub_2B48D10(v35, v34);
              v37 = v34 + 48;
              v34 += 104;
              sub_2B0BCF0(v36, v37, v38, v39, v40, v41);
              v10 = v59;
            }
            while ( v11 != v55 );
            v6 = (__int64 *)*a2;
            v12 = *((unsigned int *)a2 + 2);
            v9 = *(_QWORD *)a1;
            v13 = (__int64 *)(*a2 + v54);
          }
        }
        v14 = v10 + v9;
        v15 = v13;
        v16 = &v6[13 * v12];
        if ( v16 != v13 )
        {
          do
          {
            while ( 1 )
            {
              if ( v14 )
              {
                v17 = *v15;
                *(_QWORD *)(v14 + 8) = 0;
                *(_DWORD *)(v14 + 16) = 1;
                *(_QWORD *)v14 = v17;
                *(_DWORD *)(v14 + 20) = 0;
                v57 = v14 + 56;
                if ( v14 == -24 || (*(_QWORD *)(v14 + 24) = -4096, v14 != -40) )
                  *(_QWORD *)(v14 + 40) = -4096;
                sub_2B48D10(v14 + 8, (__int64)(v15 + 1));
                v21 = v14 + 72;
                *(_DWORD *)(v14 + 64) = 0;
                *(_QWORD *)(v14 + 56) = v14 + 72;
                *(_DWORD *)(v14 + 68) = 2;
                if ( *((_DWORD *)v15 + 16) )
                  break;
              }
              v15 += 13;
              v14 += 104LL;
              if ( v16 == v15 )
                goto LABEL_15;
            }
            v22 = (__int64)(v15 + 7);
            v15 += 13;
            v14 += 104LL;
            sub_2B0BCF0(v57, v22, v18, v21, v19, v20);
          }
          while ( v16 != v15 );
        }
LABEL_15:
        *(_DWORD *)(a1 + 8) = v56;
        v23 = *a2;
        v24 = *a2 + 104LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v24 )
        {
          do
          {
            v24 -= 104;
            v25 = *(_QWORD *)(v24 + 56);
            if ( v25 != v24 + 72 )
              _libc_free(v25);
            if ( (*(_BYTE *)(v24 + 16) & 1) == 0 )
              sub_C7D6A0(*(_QWORD *)(v24 + 24), 16LL * *(unsigned int *)(v24 + 32), 8);
          }
          while ( v23 != v24 );
        }
      }
      *((_DWORD *)a2 + 2) = 0;
    }
    else
    {
      v26 = v9 + 104 * v10;
      if ( v9 != v26 )
      {
        do
        {
          v26 -= 104LL;
          v27 = *(_QWORD *)(v26 + 56);
          if ( v27 != v26 + 72 )
            _libc_free(v27);
          if ( (*(_BYTE *)(v26 + 16) & 1) == 0 )
            sub_C7D6A0(*(_QWORD *)(v26 + 24), 16LL * *(unsigned int *)(v26 + 32), 8);
        }
        while ( v9 != v26 );
        v26 = *(_QWORD *)a1;
      }
      if ( v26 != a1 + 16 )
        _libc_free(v26);
      *(_QWORD *)a1 = *a2;
      *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
      *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
      *a2 = (__int64)v6;
      a2[1] = 0;
    }
  }
}
