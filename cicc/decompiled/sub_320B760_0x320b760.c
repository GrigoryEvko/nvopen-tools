// Function: sub_320B760
// Address: 0x320b760
//
void __fastcall sub_320B760(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rax
  __int64 *v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rcx
  __int64 v13; // r15
  char v14; // al
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // r12
  unsigned __int64 v22; // r15
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // rdi
  bool v25; // cc
  unsigned __int64 v26; // rdi
  __int64 *v27; // r15
  unsigned __int64 v28; // r14
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  __int64 *v32; // rdx
  __int64 *i; // rbx
  unsigned __int64 v34; // r14
  unsigned __int64 v35; // r15
  unsigned __int64 v36; // rdi
  __int64 v37; // r14
  __int64 v38; // r12
  __int64 v39; // r12
  unsigned __int64 v40; // r15
  unsigned __int64 v41; // rbx
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // rcx
  unsigned __int64 v47; // rdi
  __int64 *j; // rbx
  unsigned __int64 v49; // r15
  unsigned __int64 v50; // r14
  unsigned __int64 v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  int v56; // r15d
  unsigned __int64 v57; // rdi
  unsigned int v58; // [rsp+4h] [rbp-5Ch]
  __int64 v59; // [rsp+8h] [rbp-58h]
  __int64 v60; // [rsp+10h] [rbp-50h]
  __int64 v61; // [rsp+10h] [rbp-50h]
  unsigned __int64 v63[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( a1 != a2 )
  {
    v7 = *(unsigned int *)(a1 + 8);
    v8 = *(__int64 **)a1;
    v60 = a2 + 16;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v58 = *(_DWORD *)(a2 + 8);
      v59 = v58;
      if ( v58 <= v7 )
      {
        v32 = *(__int64 **)a1;
        if ( v58 )
        {
          v8 = (__int64 *)sub_320B470(v60, v60 + 88LL * v58, (__int64)v8);
          v32 = *(__int64 **)a1;
          v7 = *(unsigned int *)(a1 + 8);
        }
        for ( i = &v32[11 * v7]; v8 != i; sub_C7D6A0(i[2], 12LL * *((unsigned int *)i + 8), 4) )
        {
          i -= 11;
          if ( *((_BYTE *)i + 80) )
          {
            v25 = *((_DWORD *)i + 18) <= 0x40u;
            *((_BYTE *)i + 80) = 0;
            if ( !v25 )
            {
              v47 = i[8];
              if ( v47 )
                j_j___libc_free_0_0(v47);
            }
          }
          v34 = i[5];
          v35 = v34 + 40LL * *((unsigned int *)i + 12);
          if ( v34 != v35 )
          {
            do
            {
              v35 -= 40LL;
              v36 = *(_QWORD *)(v35 + 8);
              if ( v36 != v35 + 24 )
                _libc_free(v36);
            }
            while ( v34 != v35 );
            v34 = i[5];
          }
          if ( (__int64 *)v34 != i + 7 )
            _libc_free(v34);
        }
        *(_DWORD *)(a1 + 8) = v58;
        v37 = *(_QWORD *)a2;
        v38 = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v38 )
        {
          v39 = v38 - 88;
          if ( *(_BYTE *)(v39 + 80) )
            goto LABEL_70;
          while ( 1 )
          {
            v40 = *(_QWORD *)(v39 + 40);
            v41 = v40 + 40LL * *(unsigned int *)(v39 + 48);
            if ( v40 != v41 )
            {
              do
              {
                v41 -= 40LL;
                v42 = *(_QWORD *)(v41 + 8);
                if ( v42 != v41 + 24 )
                  _libc_free(v42);
              }
              while ( v40 != v41 );
              v40 = *(_QWORD *)(v39 + 40);
            }
            if ( v40 != v39 + 56 )
              _libc_free(v40);
            sub_C7D6A0(*(_QWORD *)(v39 + 16), 12LL * *(unsigned int *)(v39 + 32), 4);
            if ( v37 == v39 )
              break;
            v39 -= 88;
            if ( *(_BYTE *)(v39 + 80) )
            {
LABEL_70:
              v25 = *(_DWORD *)(v39 + 72) <= 0x40u;
              *(_BYTE *)(v39 + 80) = 0;
              if ( !v25 )
              {
                v43 = *(_QWORD *)(v39 + 64);
                if ( v43 )
                  j_j___libc_free_0_0(v43);
              }
            }
          }
        }
      }
      else
      {
        v9 = 11 * v7;
        if ( v58 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          for ( j = &v8[v9]; v8 != j; sub_C7D6A0(j[2], 12LL * *((unsigned int *)j + 8), 4) )
          {
            j -= 11;
            if ( *((_BYTE *)j + 80) )
            {
              v25 = *((_DWORD *)j + 18) <= 0x40u;
              *((_BYTE *)j + 80) = 0;
              if ( !v25 )
              {
                v57 = j[8];
                if ( v57 )
                  j_j___libc_free_0_0(v57);
              }
            }
            v49 = j[5];
            v50 = v49 + 40LL * *((unsigned int *)j + 12);
            if ( v49 != v50 )
            {
              do
              {
                v50 -= 40LL;
                v51 = *(_QWORD *)(v50 + 8);
                if ( v51 != v50 + 24 )
                  _libc_free(v51);
              }
              while ( v49 != v50 );
              v49 = j[5];
            }
            if ( (__int64 *)v49 != j + 7 )
              _libc_free(v49);
          }
          *(_DWORD *)(a1 + 8) = 0;
          v8 = (__int64 *)sub_C8D7D0(a1, a1 + 16, v58, 0x58u, v63, a6);
          sub_31FE0C0((__int64 **)a1, (__int64)v8, v52, v53, v54, v55);
          v56 = v63[0];
          if ( a1 + 16 != *(_QWORD *)a1 )
            _libc_free(*(_QWORD *)a1);
          *(_QWORD *)a1 = v8;
          *(_DWORD *)(a1 + 12) = v56;
          v10 = *(_QWORD *)a2;
          v60 = *(_QWORD *)a2;
          v59 = *(unsigned int *)(a2 + 8);
        }
        else
        {
          v10 = a2 + 16;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v44 = v60 + v9 * 8;
            v45 = v60;
            v61 = 88 * v7;
            sub_320B470(v45, v44, (__int64)v8);
            v46 = v61;
            v59 = *(unsigned int *)(a2 + 8);
            v60 = *(_QWORD *)a2;
            v8 = (__int64 *)(v46 + *(_QWORD *)a1);
            v10 = *(_QWORD *)a2 + v46;
          }
        }
        v11 = v10;
        v12 = v60;
        v13 = v60 + 88 * v59;
        if ( v13 != v10 )
        {
          do
          {
            if ( v8 )
            {
              v16 = *(_QWORD *)v11;
              *((_DWORD *)v8 + 8) = 0;
              v8[2] = 0;
              *((_DWORD *)v8 + 6) = 0;
              *((_DWORD *)v8 + 7) = 0;
              *v8 = v16;
              v8[1] = 1;
              v17 = *(_QWORD *)(v11 + 16);
              ++*(_QWORD *)(v11 + 8);
              v18 = v8[2];
              v8[2] = v17;
              LODWORD(v17) = *(_DWORD *)(v11 + 24);
              *(_QWORD *)(v11 + 16) = v18;
              LODWORD(v18) = *((_DWORD *)v8 + 6);
              *((_DWORD *)v8 + 6) = v17;
              LODWORD(v17) = *(_DWORD *)(v11 + 28);
              *(_DWORD *)(v11 + 24) = v18;
              LODWORD(v18) = *((_DWORD *)v8 + 7);
              *((_DWORD *)v8 + 7) = v17;
              v19 = *(unsigned int *)(v11 + 32);
              *(_DWORD *)(v11 + 28) = v18;
              LODWORD(v18) = *((_DWORD *)v8 + 8);
              *((_DWORD *)v8 + 8) = v19;
              *(_DWORD *)(v11 + 32) = v18;
              v8[5] = (__int64)(v8 + 7);
              *((_DWORD *)v8 + 12) = 0;
              *((_DWORD *)v8 + 13) = 0;
              if ( *(_DWORD *)(v11 + 48) )
                sub_31FDD40((__int64)(v8 + 5), v11 + 40, v19, v12, a5, a6);
              v14 = *(_BYTE *)(v11 + 56);
              *((_BYTE *)v8 + 80) = 0;
              *((_BYTE *)v8 + 56) = v14;
              if ( *(_BYTE *)(v11 + 80) )
              {
                *((_DWORD *)v8 + 18) = *(_DWORD *)(v11 + 72);
                v8[8] = *(_QWORD *)(v11 + 64);
                v15 = *(_BYTE *)(v11 + 76);
                *(_DWORD *)(v11 + 72) = 0;
                *((_BYTE *)v8 + 76) = v15;
                *((_BYTE *)v8 + 80) = 1;
              }
            }
            v11 += 88;
            v8 += 11;
          }
          while ( v13 != v11 );
        }
        *(_DWORD *)(a1 + 8) = v58;
        v20 = *(_QWORD *)a2;
        v21 = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v21 )
        {
          do
          {
            v21 -= 88;
            if ( *(_BYTE *)(v21 + 80) )
            {
              v25 = *(_DWORD *)(v21 + 72) <= 0x40u;
              *(_BYTE *)(v21 + 80) = 0;
              if ( !v25 )
              {
                v26 = *(_QWORD *)(v21 + 64);
                if ( v26 )
                  j_j___libc_free_0_0(v26);
              }
            }
            v22 = *(_QWORD *)(v21 + 40);
            v23 = v22 + 40LL * *(unsigned int *)(v21 + 48);
            if ( v22 != v23 )
            {
              do
              {
                v23 -= 40LL;
                v24 = *(_QWORD *)(v23 + 8);
                if ( v24 != v23 + 24 )
                  _libc_free(v24);
              }
              while ( v22 != v23 );
              v22 = *(_QWORD *)(v21 + 40);
            }
            if ( v22 != v21 + 56 )
              _libc_free(v22);
            sub_C7D6A0(*(_QWORD *)(v21 + 16), 12LL * *(unsigned int *)(v21 + 32), 4);
          }
          while ( v20 != v21 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v27 = &v8[11 * v7];
      if ( v27 != v8 )
      {
        do
        {
          v27 -= 11;
          if ( *((_BYTE *)v27 + 80) )
          {
            v25 = *((_DWORD *)v27 + 18) <= 0x40u;
            *((_BYTE *)v27 + 80) = 0;
            if ( !v25 )
            {
              v31 = v27[8];
              if ( v31 )
                j_j___libc_free_0_0(v31);
            }
          }
          v28 = v27[5];
          v29 = v28 + 40LL * *((unsigned int *)v27 + 12);
          if ( v28 != v29 )
          {
            do
            {
              v29 -= 40LL;
              v30 = *(_QWORD *)(v29 + 8);
              if ( v30 != v29 + 24 )
                _libc_free(v30);
            }
            while ( v28 != v29 );
            v28 = v27[5];
          }
          if ( (__int64 *)v28 != v27 + 7 )
            _libc_free(v28);
          sub_C7D6A0(v27[2], 12LL * *((unsigned int *)v27 + 8), 4);
        }
        while ( v27 != v8 );
        v8 = *(__int64 **)a1;
      }
      if ( v8 != (__int64 *)(a1 + 16) )
        _libc_free((unsigned __int64)v8);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v60;
    }
  }
}
