// Function: sub_145E880
// Address: 0x145e880
//
void __fastcall sub_145E880(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  unsigned __int64 v3; // rdx
  _QWORD *v4; // r13
  _QWORD *v5; // rbx
  __int64 v6; // r15
  _QWORD *v7; // rax
  _QWORD *v8; // rbx
  _QWORD *i; // rcx
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r14
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  _QWORD *v17; // r15
  __int64 v18; // r14
  __int64 v19; // rdx
  _QWORD *v20; // rax
  _QWORD *v21; // rcx
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  _QWORD *v24; // rax
  _QWORD *j; // r13
  __int64 v26; // r15
  __int64 v27; // rax
  _QWORD *v28; // rbx
  _QWORD *v29; // rdx
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  __int64 v32; // r13
  __int64 v33; // r15
  __int64 v34; // rax
  _QWORD *v35; // rbx
  _QWORD *v36; // r14
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rax
  _QWORD *v42; // r15
  _QWORD *v43; // rcx
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdx
  _QWORD *v47; // r13
  __int64 v48; // r14
  __int64 v49; // rdx
  _QWORD *v50; // rax
  _QWORD *v51; // rcx
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rbx
  __int64 v56; // rax
  _QWORD *v57; // r15
  _QWORD *v58; // rdx
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rdi
  unsigned __int64 v61; // [rsp+0h] [rbp-60h]
  _QWORD *v62; // [rsp+8h] [rbp-58h]
  unsigned int v63; // [rsp+14h] [rbp-4Ch]
  _QWORD *v65; // [rsp+20h] [rbp-40h]
  unsigned __int64 v66; // [rsp+20h] [rbp-40h]
  _QWORD *v67; // [rsp+20h] [rbp-40h]
  _QWORD *v68; // [rsp+20h] [rbp-40h]
  _QWORD *v69; // [rsp+28h] [rbp-38h]
  _QWORD *v70; // [rsp+28h] [rbp-38h]
  _QWORD *v71; // [rsp+28h] [rbp-38h]
  _QWORD *v72; // [rsp+28h] [rbp-38h]
  _QWORD *v73; // [rsp+28h] [rbp-38h]
  unsigned __int64 v74; // [rsp+28h] [rbp-38h]
  unsigned __int64 v75; // [rsp+28h] [rbp-38h]
  _QWORD *v76; // [rsp+28h] [rbp-38h]
  _QWORD *v77; // [rsp+28h] [rbp-38h]

  if ( a1 != a2 )
  {
    v2 = *(_QWORD **)a1;
    v3 = *(unsigned int *)(a1 + 8);
    v4 = (_QWORD *)(a2 + 16);
    v5 = *(_QWORD **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v63 = *(_DWORD *)(a2 + 8);
      v6 = v63;
      if ( v63 > v3 )
      {
        if ( v63 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v47 = &v2[3 * v3];
          while ( v47 != v5 )
          {
            while ( 1 )
            {
              v48 = *(v47 - 1);
              v47 -= 3;
              if ( !v48 )
                break;
              v49 = *(unsigned int *)(v48 + 208);
              *(_QWORD *)v48 = &unk_49EC708;
              if ( (_DWORD)v49 )
              {
                v50 = *(_QWORD **)(v48 + 192);
                v51 = &v50[7 * v49];
                do
                {
                  if ( *v50 != -16 && *v50 != -8 )
                  {
                    v52 = v50[1];
                    if ( (_QWORD *)v52 != v50 + 3 )
                    {
                      v67 = v51;
                      v76 = v50;
                      _libc_free(v52);
                      v51 = v67;
                      v50 = v76;
                    }
                  }
                  v50 += 7;
                }
                while ( v51 != v50 );
              }
              j___libc_free_0(*(_QWORD *)(v48 + 192));
              v53 = *(_QWORD *)(v48 + 40);
              if ( v53 != v48 + 56 )
                _libc_free(v53);
              j_j___libc_free_0(v48, 216);
              if ( v47 == v5 )
                goto LABEL_97;
            }
          }
LABEL_97:
          *(_DWORD *)(a1 + 8) = 0;
          sub_145E660(a1, v63);
          v4 = *(_QWORD **)a2;
          v6 = *(unsigned int *)(a2 + 8);
          v3 = 0;
          v5 = *(_QWORD **)a1;
          v7 = *(_QWORD **)a2;
        }
        else
        {
          v7 = (_QWORD *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v3 *= 24LL;
            v61 = v3;
            v62 = (_QWORD *)((char *)v2 + v3);
            do
            {
              *v2 = *v4;
              v2[1] = v4[1];
              v39 = v4[2];
              v4[2] = 0;
              v40 = v2[2];
              v2[2] = v39;
              if ( v40 )
              {
                *(_QWORD *)v40 = &unk_49EC708;
                v41 = *(unsigned int *)(v40 + 208);
                if ( (_DWORD)v41 )
                {
                  v42 = *(_QWORD **)(v40 + 192);
                  v43 = &v42[7 * v41];
                  do
                  {
                    if ( *v42 != -16 && *v42 != -8 )
                    {
                      v44 = v42[1];
                      if ( (_QWORD *)v44 != v42 + 3 )
                      {
                        v66 = v3;
                        v73 = v43;
                        _libc_free(v44);
                        v3 = v66;
                        v43 = v73;
                      }
                    }
                    v42 += 7;
                  }
                  while ( v43 != v42 );
                }
                v74 = v3;
                j___libc_free_0(*(_QWORD *)(v40 + 192));
                v45 = *(_QWORD *)(v40 + 40);
                v46 = v74;
                if ( v45 != v40 + 56 )
                {
                  _libc_free(v45);
                  v46 = v74;
                }
                v75 = v46;
                j_j___libc_free_0(v40, 216);
                v3 = v75;
              }
              v4 += 3;
              v2 += 3;
            }
            while ( v2 != v62 );
            v4 = *(_QWORD **)a2;
            v6 = *(unsigned int *)(a2 + 8);
            v7 = (_QWORD *)(*(_QWORD *)a2 + v61);
            v5 = *(_QWORD **)a1;
          }
        }
        v8 = (_QWORD *)((char *)v5 + v3);
        for ( i = &v4[3 * v6]; i != v7; v8 += 3 )
        {
          if ( v8 )
          {
            *v8 = *v7;
            v8[1] = v7[1];
            v8[2] = v7[2];
            v7[2] = 0;
          }
          v7 += 3;
        }
        *(_DWORD *)(a1 + 8) = v63;
        v10 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
        v69 = *(_QWORD **)a2;
        while ( v69 != (_QWORD *)v10 )
        {
          v11 = *(_QWORD *)(v10 - 8);
          v10 -= 24;
          if ( v11 )
          {
            *(_QWORD *)v11 = &unk_49EC708;
            v12 = *(unsigned int *)(v11 + 208);
            if ( (_DWORD)v12 )
            {
              v13 = *(_QWORD **)(v11 + 192);
              v14 = &v13[7 * v12];
              do
              {
                if ( *v13 != -8 && *v13 != -16 )
                {
                  v15 = v13[1];
                  if ( (_QWORD *)v15 != v13 + 3 )
                    _libc_free(v15);
                }
                v13 += 7;
              }
              while ( v14 != v13 );
            }
            j___libc_free_0(*(_QWORD *)(v11 + 192));
            v16 = *(_QWORD *)(v11 + 40);
            if ( v16 != v11 + 56 )
              _libc_free(v16);
            j_j___libc_free_0(v11, 216);
          }
        }
        goto LABEL_23;
      }
      v24 = *(_QWORD **)a1;
      if ( v63 )
      {
        v68 = &v2[3 * v63];
        do
        {
          *v2 = *v4;
          v2[1] = v4[1];
          v54 = v4[2];
          v4[2] = 0;
          v55 = v2[2];
          v2[2] = v54;
          if ( v55 )
          {
            *(_QWORD *)v55 = &unk_49EC708;
            v56 = *(unsigned int *)(v55 + 208);
            if ( (_DWORD)v56 )
            {
              v57 = *(_QWORD **)(v55 + 192);
              v58 = &v57[7 * v56];
              do
              {
                if ( *v57 != -8 && *v57 != -16 )
                {
                  v59 = v57[1];
                  if ( (_QWORD *)v59 != v57 + 3 )
                  {
                    v77 = v58;
                    _libc_free(v59);
                    v58 = v77;
                  }
                }
                v57 += 7;
              }
              while ( v58 != v57 );
            }
            j___libc_free_0(*(_QWORD *)(v55 + 192));
            v60 = *(_QWORD *)(v55 + 40);
            if ( v60 != v55 + 56 )
              _libc_free(v60);
            j_j___libc_free_0(v55, 216);
          }
          v4 += 3;
          v2 += 3;
        }
        while ( v2 != v68 );
        v24 = *(_QWORD **)a1;
        v3 = *(unsigned int *)(a1 + 8);
      }
      for ( j = &v24[3 * v3]; v2 != j; j -= 3 )
      {
        v26 = *(j - 1);
        if ( v26 )
        {
          *(_QWORD *)v26 = &unk_49EC708;
          v27 = *(unsigned int *)(v26 + 208);
          if ( (_DWORD)v27 )
          {
            v28 = *(_QWORD **)(v26 + 192);
            v29 = &v28[7 * v27];
            do
            {
              if ( *v28 != -16 && *v28 != -8 )
              {
                v30 = v28[1];
                if ( (_QWORD *)v30 != v28 + 3 )
                {
                  v71 = v29;
                  _libc_free(v30);
                  v29 = v71;
                }
              }
              v28 += 7;
            }
            while ( v29 != v28 );
          }
          j___libc_free_0(*(_QWORD *)(v26 + 192));
          v31 = *(_QWORD *)(v26 + 40);
          if ( v31 != v26 + 56 )
            _libc_free(v31);
          j_j___libc_free_0(v26, 216);
        }
      }
      *(_DWORD *)(a1 + 8) = v63;
      v32 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
      v72 = *(_QWORD **)a2;
      if ( *(_QWORD *)a2 == v32 )
      {
LABEL_23:
        *(_DWORD *)(a2 + 8) = 0;
        return;
      }
      do
      {
        v33 = *(_QWORD *)(v32 - 8);
        v32 -= 24;
        if ( v33 )
        {
          *(_QWORD *)v33 = &unk_49EC708;
          v34 = *(unsigned int *)(v33 + 208);
          if ( (_DWORD)v34 )
          {
            v35 = *(_QWORD **)(v33 + 192);
            v36 = &v35[7 * v34];
            do
            {
              if ( *v35 != -8 && *v35 != -16 )
              {
                v37 = v35[1];
                if ( (_QWORD *)v37 != v35 + 3 )
                  _libc_free(v37);
              }
              v35 += 7;
            }
            while ( v36 != v35 );
          }
          j___libc_free_0(*(_QWORD *)(v33 + 192));
          v38 = *(_QWORD *)(v33 + 40);
          if ( v38 != v33 + 56 )
            _libc_free(v38);
          j_j___libc_free_0(v33, 216);
        }
      }
      while ( v72 != (_QWORD *)v32 );
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v17 = &v2[3 * v3];
      if ( v17 != v2 )
      {
        do
        {
          v18 = *(v17 - 1);
          v17 -= 3;
          if ( v18 )
          {
            v19 = *(unsigned int *)(v18 + 208);
            *(_QWORD *)v18 = &unk_49EC708;
            if ( (_DWORD)v19 )
            {
              v20 = *(_QWORD **)(v18 + 192);
              v21 = &v20[7 * v19];
              do
              {
                if ( *v20 != -16 && *v20 != -8 )
                {
                  v22 = v20[1];
                  if ( (_QWORD *)v22 != v20 + 3 )
                  {
                    v65 = v20;
                    v70 = v21;
                    _libc_free(v22);
                    v20 = v65;
                    v21 = v70;
                  }
                }
                v20 += 7;
              }
              while ( v21 != v20 );
            }
            j___libc_free_0(*(_QWORD *)(v18 + 192));
            v23 = *(_QWORD *)(v18 + 40);
            if ( v23 != v18 + 56 )
              _libc_free(v23);
            j_j___libc_free_0(v18, 216);
          }
        }
        while ( v17 != v5 );
        v2 = *(_QWORD **)a1;
      }
      if ( v2 != (_QWORD *)(a1 + 16) )
        _libc_free((unsigned __int64)v2);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v4;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
