// Function: sub_15ECD30
// Address: 0x15ecd30
//
void __fastcall sub_15ECD30(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // rax
  unsigned __int64 v8; // r14
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned __int64 i; // rbx
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rdx
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // r12
  _QWORD *v21; // r15
  unsigned __int64 v22; // r12
  _QWORD *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rdx
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // r12
  _QWORD *v30; // r15
  unsigned __int64 v31; // r13
  _QWORD *v32; // rbx
  __int64 v33; // r14
  __int64 *v34; // rbx
  __int64 v35; // r13
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // r12
  __int64 *v43; // r13
  __int64 v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned int v49; // [rsp+4h] [rbp-5Ch]
  __int64 v50; // [rsp+10h] [rbp-50h]
  __int64 v52; // [rsp+20h] [rbp-40h]
  __int64 v53; // [rsp+28h] [rbp-38h]

  if ( (_DWORD *)a1 != a2 )
  {
    v8 = *(unsigned int *)(a1 + 8);
    v49 = a2[2];
    v6 = v49;
    v50 = v49;
    v53 = *(_QWORD *)a1;
    v52 = *(_QWORD *)a1;
    if ( v49 <= v8 )
    {
      v7 = *(_QWORD *)a1;
      v14 = *(_QWORD *)a1;
      if ( v49 )
      {
        v33 = v7 + 16;
        v34 = (__int64 *)(*(_QWORD *)a2 + 16LL);
        v35 = v7 + 16 + 192LL * v49;
        do
        {
          *(_DWORD *)(v33 - 16) = *((_DWORD *)v34 - 4);
          *(_DWORD *)(v33 - 12) = *((_DWORD *)v34 - 3);
          *(_BYTE *)(v33 - 8) = *((_BYTE *)v34 - 8);
          *(_BYTE *)(v33 - 7) = *((_BYTE *)v34 - 7);
          *(_BYTE *)(v33 - 6) = *((_BYTE *)v34 - 6);
          *(_BYTE *)(v33 - 5) = *((_BYTE *)v34 - 5);
          v36 = *((unsigned int *)v34 - 1);
          *(_DWORD *)(v33 - 4) = v36;
          sub_15EB610(v33, v34, v36, v6, a5, a6);
          v37 = v33 + 48;
          v33 += 192;
          sub_15EC730(v37, (__int64)(v34 + 6), v38, v39, v40, v41);
          v34 += 24;
        }
        while ( v35 != v33 );
        v52 = v53 + 192LL * v49;
        v14 = *(_QWORD *)a1;
        v8 = *(unsigned int *)(a1 + 8);
      }
      v15 = v14 + 192 * v8;
      while ( v52 != v15 )
      {
        v16 = *(unsigned int *)(v15 - 120);
        v17 = *(_QWORD *)(v15 - 128);
        v15 -= 192;
        v18 = v17 + 56 * v16;
        if ( v17 != v18 )
        {
          do
          {
            v19 = *(unsigned int *)(v18 - 40);
            v20 = *(_QWORD *)(v18 - 48);
            v18 -= 56LL;
            v19 *= 32;
            v21 = (_QWORD *)(v20 + v19);
            if ( v20 != v20 + v19 )
            {
              do
              {
                v21 -= 4;
                if ( (_QWORD *)*v21 != v21 + 2 )
                  j_j___libc_free_0(*v21, v21[2] + 1LL);
              }
              while ( (_QWORD *)v20 != v21 );
              v20 = *(_QWORD *)(v18 + 8);
            }
            if ( v20 != v18 + 24 )
              _libc_free(v20);
          }
          while ( v17 != v18 );
          v17 = *(_QWORD *)(v15 + 64);
        }
        if ( v17 != v15 + 80 )
          _libc_free(v17);
        v22 = *(_QWORD *)(v15 + 16);
        v23 = (_QWORD *)(v22 + 32LL * *(unsigned int *)(v15 + 24));
        if ( (_QWORD *)v22 != v23 )
        {
          do
          {
            v23 -= 4;
            if ( (_QWORD *)*v23 != v23 + 2 )
              j_j___libc_free_0(*v23, v23[2] + 1LL);
          }
          while ( (_QWORD *)v22 != v23 );
          v22 = *(_QWORD *)(v15 + 16);
        }
        if ( v22 != v15 + 32 )
          _libc_free(v22);
      }
    }
    else
    {
      if ( v49 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v24 = v53 + 192 * v8;
        while ( v53 != v24 )
        {
          v25 = *(unsigned int *)(v24 - 120);
          v26 = *(_QWORD *)(v24 - 128);
          v24 -= 192;
          v27 = v26 + 56 * v25;
          if ( v26 != v27 )
          {
            do
            {
              v28 = *(unsigned int *)(v27 - 40);
              v29 = *(_QWORD *)(v27 - 48);
              v27 -= 56LL;
              v28 *= 32;
              v30 = (_QWORD *)(v29 + v28);
              if ( v29 != v29 + v28 )
              {
                do
                {
                  v30 -= 4;
                  if ( (_QWORD *)*v30 != v30 + 2 )
                    j_j___libc_free_0(*v30, v30[2] + 1LL);
                }
                while ( (_QWORD *)v29 != v30 );
                v29 = *(_QWORD *)(v27 + 8);
              }
              if ( v29 != v27 + 24 )
                _libc_free(v29);
            }
            while ( v26 != v27 );
            v26 = *(_QWORD *)(v24 + 64);
          }
          if ( v26 != v24 + 80 )
            _libc_free(v26);
          v31 = *(_QWORD *)(v24 + 16);
          v32 = (_QWORD *)(v31 + 32LL * *(unsigned int *)(v24 + 24));
          if ( (_QWORD *)v31 != v32 )
          {
            do
            {
              v32 -= 4;
              if ( (_QWORD *)*v32 != v32 + 2 )
                j_j___libc_free_0(*v32, v32[2] + 1LL);
            }
            while ( (_QWORD *)v31 != v32 );
            v31 = *(_QWORD *)(v24 + 16);
          }
          if ( v31 != v24 + 32 )
            _libc_free(v31);
        }
        v8 = 0;
        *(_DWORD *)(a1 + 8) = 0;
        sub_15ECA10(a1, v49);
        v50 = (unsigned int)a2[2];
        v53 = *(_QWORD *)a1;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v42 = v53 + 16;
        v8 *= 192LL;
        v43 = (__int64 *)(*(_QWORD *)a2 + 16LL);
        do
        {
          *(_DWORD *)(v42 - 16) = *((_DWORD *)v43 - 4);
          *(_DWORD *)(v42 - 12) = *((_DWORD *)v43 - 3);
          *(_BYTE *)(v42 - 8) = *((_BYTE *)v43 - 8);
          *(_BYTE *)(v42 - 7) = *((_BYTE *)v43 - 7);
          *(_BYTE *)(v42 - 6) = *((_BYTE *)v43 - 6);
          *(_BYTE *)(v42 - 5) = *((_BYTE *)v43 - 5);
          *(_DWORD *)(v42 - 4) = *((_DWORD *)v43 - 1);
          sub_15EB610(v42, v43, a3, v6, a5, a6);
          v44 = v42 + 48;
          v42 += 192;
          sub_15EC730(v44, (__int64)(v43 + 6), v45, v46, v47, v48);
          v43 += 24;
        }
        while ( v42 != v53 + 16 + v8 );
        v50 = (unsigned int)a2[2];
        v53 = *(_QWORD *)a1;
      }
      v9 = v8 + v53;
      v10 = *(_QWORD *)a2 + 192 * v50;
      for ( i = v8 + *(_QWORD *)a2; v10 != i; v9 += 192 )
      {
        if ( v9 )
        {
          *(_DWORD *)v9 = *(_DWORD *)i;
          *(_DWORD *)(v9 + 4) = *(_DWORD *)(i + 4);
          *(_BYTE *)(v9 + 8) = *(_BYTE *)(i + 8);
          *(_BYTE *)(v9 + 9) = *(_BYTE *)(i + 9);
          *(_BYTE *)(v9 + 10) = *(_BYTE *)(i + 10);
          *(_BYTE *)(v9 + 11) = *(_BYTE *)(i + 11);
          v12 = *(_DWORD *)(i + 12);
          *(_DWORD *)(v9 + 24) = 0;
          *(_DWORD *)(v9 + 12) = v12;
          *(_QWORD *)(v9 + 16) = v9 + 32;
          *(_DWORD *)(v9 + 28) = 1;
          v13 = *(unsigned int *)(i + 24);
          if ( (_DWORD)v13 )
            sub_15EB610(v9 + 16, (__int64 *)(i + 16), v13, v6, a5, a6);
          *(_DWORD *)(v9 + 72) = 0;
          *(_QWORD *)(v9 + 64) = v9 + 80;
          *(_DWORD *)(v9 + 76) = 2;
          if ( *(_DWORD *)(i + 72) )
            sub_15EC730(v9 + 64, i + 64, v13, v6, a5, a6);
        }
        i += 192LL;
      }
    }
    *(_DWORD *)(a1 + 8) = v49;
  }
}
