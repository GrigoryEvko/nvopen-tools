// Function: sub_2757E90
// Address: 0x2757e90
//
void __fastcall sub_2757E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  __int64 v10; // r15
  unsigned __int64 v11; // r12
  __int64 v12; // rbx
  char v13; // al
  __int64 *v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // rbx
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rbx
  __int64 v23; // r15
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // r15
  unsigned __int64 v26; // rdi
  __int64 v27; // rdx
  unsigned __int64 v28; // rbx
  __int64 v29; // r12
  unsigned __int64 v30; // r15
  unsigned __int64 v31; // r12
  unsigned __int64 v32; // rdi
  __int64 v33; // r14
  __int64 v34; // r12
  __int64 v35; // rbx
  unsigned __int64 v36; // r15
  unsigned __int64 v37; // rbx
  unsigned __int64 v38; // rdi
  __int64 v39; // r12
  __int64 *v40; // r15
  unsigned __int64 v41; // rbx
  int v42; // ecx
  __int64 *v43; // rsi
  __int64 v44; // rdi
  __int64 v45; // rdx
  unsigned __int64 v46; // r15
  __int64 v47; // rax
  unsigned __int64 v48; // rbx
  __int64 v49; // r12
  unsigned __int64 v50; // rdi
  __int64 *v51; // rbx
  __int64 v52; // r15
  int v53; // ecx
  __int64 *v54; // rsi
  __int64 v55; // rdi
  unsigned int v56; // [rsp-5Ch] [rbp-5Ch]
  __int64 v57; // [rsp-58h] [rbp-58h]
  __int64 v58; // [rsp-50h] [rbp-50h]
  unsigned __int64 v59; // [rsp-48h] [rbp-48h]
  __int64 v60; // [rsp-48h] [rbp-48h]
  __int64 v61; // [rsp-40h] [rbp-40h]
  unsigned __int64 v62; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v8 = *(_QWORD *)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v58 = a2 + 16;
    v61 = *(_QWORD *)a1;
    v59 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v56 = *(_DWORD *)(a2 + 8);
      v57 = v56;
      if ( v56 <= v9 )
      {
        v27 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v51 = (__int64 *)(a2 + 24);
          v52 = v61 + 8;
          do
          {
            v53 = *((_DWORD *)v51 - 2);
            v54 = v51;
            v55 = v52;
            v51 += 11;
            v52 += 88;
            *(_DWORD *)(v52 - 96) = v53;
            *(_BYTE *)(v52 - 92) = *((_BYTE *)v51 - 92);
            sub_27578C0(v55, v54);
          }
          while ( v52 != v61 + 8 + 88LL * v56 );
          v27 = *(_QWORD *)a1;
          v59 = v61 + 88LL * v56;
          v9 = *(unsigned int *)(a1 + 8);
        }
        v28 = v27 + 88 * v9;
        while ( v59 != v28 )
        {
          v29 = *(unsigned int *)(v28 - 72);
          v30 = *(_QWORD *)(v28 - 80);
          v28 -= 88LL;
          v31 = v30 + 32 * v29;
          if ( v30 != v31 )
          {
            do
            {
              v31 -= 32LL;
              if ( *(_DWORD *)(v31 + 24) > 0x40u )
              {
                v32 = *(_QWORD *)(v31 + 16);
                if ( v32 )
                  j_j___libc_free_0_0(v32);
              }
              if ( *(_DWORD *)(v31 + 8) > 0x40u && *(_QWORD *)v31 )
                j_j___libc_free_0_0(*(_QWORD *)v31);
            }
            while ( v30 != v31 );
            v30 = *(_QWORD *)(v28 + 8);
          }
          if ( v30 != v28 + 24 )
            _libc_free(v30);
        }
        *(_DWORD *)(a1 + 8) = v56;
        v33 = *(_QWORD *)a2;
        v34 = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v34 )
        {
          do
          {
            v35 = *(unsigned int *)(v34 - 72);
            v36 = *(_QWORD *)(v34 - 80);
            v34 -= 88;
            v37 = v36 + 32 * v35;
            if ( v36 != v37 )
            {
              do
              {
                v37 -= 32LL;
                if ( *(_DWORD *)(v37 + 24) > 0x40u )
                {
                  v38 = *(_QWORD *)(v37 + 16);
                  if ( v38 )
                    j_j___libc_free_0_0(v38);
                }
                if ( *(_DWORD *)(v37 + 8) > 0x40u && *(_QWORD *)v37 )
                  j_j___libc_free_0_0(*(_QWORD *)v37);
              }
              while ( v36 != v37 );
              v36 = *(_QWORD *)(v34 + 8);
            }
            if ( v36 != v34 + 24 )
              _libc_free(v36);
          }
          while ( v33 != v34 );
        }
      }
      else
      {
        if ( v56 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v45 = *(_QWORD *)a1;
          v46 = v61 + 88 * v9;
          while ( v46 != v61 )
          {
            v47 = *(unsigned int *)(v46 - 72);
            v48 = *(_QWORD *)(v46 - 80);
            v46 -= 88LL;
            v47 *= 32;
            v49 = v48 + v47;
            if ( v48 != v48 + v47 )
            {
              do
              {
                v49 -= 32;
                if ( *(_DWORD *)(v49 + 24) > 0x40u )
                {
                  v50 = *(_QWORD *)(v49 + 16);
                  if ( v50 )
                    j_j___libc_free_0_0(v50);
                }
                if ( *(_DWORD *)(v49 + 8) > 0x40u && *(_QWORD *)v49 )
                  j_j___libc_free_0_0(*(_QWORD *)v49);
              }
              while ( v48 != v49 );
              v48 = *(_QWORD *)(v46 + 8);
            }
            if ( v48 != v46 + 24 )
              _libc_free(v48);
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_2757D10(a1, v56, v45, v8, a5, a6);
          v10 = *(_QWORD *)a2;
          v57 = *(unsigned int *)(a2 + 8);
          v9 = 0;
          v58 = *(_QWORD *)a2;
          v61 = *(_QWORD *)a1;
        }
        else
        {
          v10 = a2 + 16;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v39 = v61 + 8;
            v40 = (__int64 *)(a2 + 24);
            v60 = 88 * v9;
            v9 *= 88LL;
            v41 = v61 + 8 + v9;
            do
            {
              v42 = *((_DWORD *)v40 - 2);
              v43 = v40;
              v44 = v39;
              v39 += 88;
              v62 = v9;
              v40 += 11;
              *(_DWORD *)(v39 - 96) = v42;
              *(_BYTE *)(v39 - 92) = *((_BYTE *)v40 - 92);
              sub_27578C0(v44, v43);
              v9 = v62;
            }
            while ( v41 != v39 );
            v58 = *(_QWORD *)a2;
            v10 = *(_QWORD *)a2 + v60;
            v57 = *(unsigned int *)(a2 + 8);
            v61 = *(_QWORD *)a1;
          }
        }
        v11 = v9 + v61;
        v12 = v58 + 88 * v57;
        while ( v12 != v10 )
        {
          while ( 1 )
          {
            if ( v11 )
            {
              *(_DWORD *)v11 = *(_DWORD *)v10;
              v13 = *(_BYTE *)(v10 + 4);
              *(_DWORD *)(v11 + 16) = 0;
              *(_BYTE *)(v11 + 4) = v13;
              *(_QWORD *)(v11 + 8) = v11 + 24;
              *(_DWORD *)(v11 + 20) = 2;
              if ( *(_DWORD *)(v10 + 16) )
                break;
            }
            v10 += 88;
            v11 += 88LL;
            if ( v12 == v10 )
              goto LABEL_12;
          }
          v14 = (__int64 *)(v10 + 8);
          v15 = v11 + 8;
          v10 += 88;
          v11 += 88LL;
          sub_27578C0(v15, v14);
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v56;
        v16 = *(_QWORD *)a2;
        v17 = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v17 )
        {
          do
          {
            v18 = *(unsigned int *)(v17 - 72);
            v19 = *(_QWORD *)(v17 - 80);
            v17 -= 88;
            v20 = v19 + 32 * v18;
            if ( v19 != v20 )
            {
              do
              {
                v20 -= 32LL;
                if ( *(_DWORD *)(v20 + 24) > 0x40u )
                {
                  v21 = *(_QWORD *)(v20 + 16);
                  if ( v21 )
                    j_j___libc_free_0_0(v21);
                }
                if ( *(_DWORD *)(v20 + 8) > 0x40u && *(_QWORD *)v20 )
                  j_j___libc_free_0_0(*(_QWORD *)v20);
              }
              while ( v19 != v20 );
              v19 = *(_QWORD *)(v17 + 8);
            }
            if ( v19 != v17 + 24 )
              _libc_free(v19);
          }
          while ( v16 != v17 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v22 = v8 + 88 * v9;
      if ( v22 != v8 )
      {
        do
        {
          v23 = *(unsigned int *)(v22 - 72);
          v24 = *(_QWORD *)(v22 - 80);
          v22 -= 88LL;
          v25 = v24 + 32 * v23;
          if ( v24 != v25 )
          {
            do
            {
              v25 -= 32LL;
              if ( *(_DWORD *)(v25 + 24) > 0x40u )
              {
                v26 = *(_QWORD *)(v25 + 16);
                if ( v26 )
                  j_j___libc_free_0_0(v26);
              }
              if ( *(_DWORD *)(v25 + 8) > 0x40u && *(_QWORD *)v25 )
                j_j___libc_free_0_0(*(_QWORD *)v25);
            }
            while ( v24 != v25 );
            v24 = *(_QWORD *)(v22 + 8);
          }
          if ( v24 != v22 + 24 )
            _libc_free(v24);
        }
        while ( v22 != v61 );
        v59 = *(_QWORD *)a1;
      }
      if ( v59 != a1 + 16 )
        _libc_free(v59);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v58;
    }
  }
}
