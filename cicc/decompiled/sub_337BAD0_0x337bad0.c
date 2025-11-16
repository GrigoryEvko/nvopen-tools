// Function: sub_337BAD0
// Address: 0x337bad0
//
void __fastcall sub_337BAD0(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  int *v5; // r15
  unsigned __int64 v6; // r12
  int *v7; // rbx
  int v8; // eax
  __int64 v9; // rsi
  __int64 v10; // rdi
  int *v11; // r14
  __int64 v12; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // r15
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // rbx
  __int64 v17; // r15
  unsigned __int64 v18; // r12
  unsigned __int64 *v19; // r15
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rbx
  __int64 v22; // r12
  unsigned __int64 v23; // r15
  unsigned __int64 *v24; // r12
  int *v25; // r14
  __int64 v26; // r12
  __int64 v27; // rbx
  unsigned __int64 v28; // r15
  unsigned __int64 *v29; // rbx
  __int64 v30; // r12
  __int64 v31; // r15
  unsigned __int64 v32; // rbx
  int v33; // ecx
  __int64 v34; // rsi
  __int64 v35; // rdi
  unsigned __int64 v36; // r15
  __int64 v37; // rax
  unsigned __int64 v38; // rbx
  unsigned __int64 *v39; // r12
  __int64 v40; // rbx
  __int64 v41; // r15
  int v42; // ecx
  __int64 v43; // rsi
  __int64 v44; // rdi
  unsigned int v45; // [rsp-5Ch] [rbp-5Ch]
  __int64 v46; // [rsp-58h] [rbp-58h]
  int *v47; // [rsp-50h] [rbp-50h]
  unsigned __int64 v48; // [rsp-48h] [rbp-48h]
  __int64 v49; // [rsp-48h] [rbp-48h]
  unsigned __int64 v50; // [rsp-40h] [rbp-40h]
  unsigned __int64 v51; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v4 = *(unsigned int *)(a1 + 8);
    v47 = (int *)(a2 + 16);
    v50 = *(_QWORD *)a1;
    v48 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v45 = *(_DWORD *)(a2 + 8);
      v46 = v45;
      if ( v45 <= v4 )
      {
        v20 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v40 = a2 + 24;
          v41 = v50 + 8;
          do
          {
            v42 = *(_DWORD *)(v40 - 8);
            v43 = v40;
            v44 = v41;
            v40 += 56;
            v41 += 56;
            *(_DWORD *)(v41 - 64) = v42;
            sub_337B320(v44, v43);
          }
          while ( v50 + 8 + 56LL * v45 != v41 );
          v20 = *(_QWORD *)a1;
          v48 = v50 + 56LL * v45;
          v4 = *(unsigned int *)(a1 + 8);
        }
        v21 = v20 + 56 * v4;
        while ( v48 != v21 )
        {
          v22 = *(unsigned int *)(v21 - 40);
          v23 = *(_QWORD *)(v21 - 48);
          v21 -= 56LL;
          v24 = (unsigned __int64 *)(v23 + 32 * v22);
          if ( (unsigned __int64 *)v23 != v24 )
          {
            do
            {
              v24 -= 4;
              if ( (unsigned __int64 *)*v24 != v24 + 2 )
                j_j___libc_free_0(*v24);
            }
            while ( (unsigned __int64 *)v23 != v24 );
            v23 = *(_QWORD *)(v21 + 8);
          }
          if ( v23 != v21 + 24 )
            _libc_free(v23);
        }
        *(_DWORD *)(a1 + 8) = v45;
        v25 = *(int **)a2;
        v26 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v26 )
        {
          do
          {
            v27 = *(unsigned int *)(v26 - 40);
            v28 = *(_QWORD *)(v26 - 48);
            v26 -= 56;
            v29 = (unsigned __int64 *)(v28 + 32 * v27);
            if ( (unsigned __int64 *)v28 != v29 )
            {
              do
              {
                v29 -= 4;
                if ( (unsigned __int64 *)*v29 != v29 + 2 )
                  j_j___libc_free_0(*v29);
              }
              while ( (unsigned __int64 *)v28 != v29 );
              v28 = *(_QWORD *)(v26 + 8);
            }
            if ( v28 != v26 + 24 )
              _libc_free(v28);
          }
          while ( v25 != (int *)v26 );
        }
      }
      else
      {
        if ( v45 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v36 = v50 + 56 * v4;
          while ( v50 != v36 )
          {
            v37 = *(unsigned int *)(v36 - 40);
            v38 = *(_QWORD *)(v36 - 48);
            v36 -= 56LL;
            v37 *= 32;
            v39 = (unsigned __int64 *)(v38 + v37);
            if ( v38 != v38 + v37 )
            {
              do
              {
                v39 -= 4;
                if ( (unsigned __int64 *)*v39 != v39 + 2 )
                  j_j___libc_free_0(*v39);
              }
              while ( (unsigned __int64 *)v38 != v39 );
              v38 = *(_QWORD *)(v36 + 8);
            }
            if ( v38 != v36 + 24 )
              _libc_free(v38);
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_B3C890(a1, v45);
          v5 = *(int **)a2;
          v46 = *(unsigned int *)(a2 + 8);
          v4 = 0;
          v47 = *(int **)a2;
          v50 = *(_QWORD *)a1;
        }
        else
        {
          v5 = (int *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v30 = v50 + 8;
            v31 = a2 + 24;
            v49 = 56 * v4;
            v4 *= 56LL;
            v32 = v50 + 8 + v4;
            do
            {
              v33 = *(_DWORD *)(v31 - 8);
              v34 = v31;
              v35 = v30;
              v30 += 56;
              v51 = v4;
              v31 += 56;
              *(_DWORD *)(v30 - 64) = v33;
              sub_337B320(v35, v34);
              v4 = v51;
            }
            while ( v32 != v30 );
            v47 = *(int **)a2;
            v5 = (int *)(*(_QWORD *)a2 + v49);
            v46 = *(unsigned int *)(a2 + 8);
            v50 = *(_QWORD *)a1;
          }
        }
        v6 = v4 + v50;
        v7 = &v47[14 * v46];
        while ( v7 != v5 )
        {
          while ( 1 )
          {
            if ( v6 )
            {
              v8 = *v5;
              *(_DWORD *)(v6 + 16) = 0;
              *(_DWORD *)(v6 + 20) = 1;
              *(_DWORD *)v6 = v8;
              *(_QWORD *)(v6 + 8) = v6 + 24;
              if ( v5[4] )
                break;
            }
            v5 += 14;
            v6 += 56LL;
            if ( v7 == v5 )
              goto LABEL_12;
          }
          v9 = (__int64)(v5 + 2);
          v10 = v6 + 8;
          v5 += 14;
          v6 += 56LL;
          sub_337B320(v10, v9);
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v45;
        v11 = *(int **)a2;
        v12 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v12 )
        {
          do
          {
            v13 = *(unsigned int *)(v12 - 40);
            v14 = *(_QWORD *)(v12 - 48);
            v12 -= 56;
            v15 = (unsigned __int64 *)(v14 + 32 * v13);
            if ( (unsigned __int64 *)v14 != v15 )
            {
              do
              {
                v15 -= 4;
                if ( (unsigned __int64 *)*v15 != v15 + 2 )
                  j_j___libc_free_0(*v15);
              }
              while ( (unsigned __int64 *)v14 != v15 );
              v14 = *(_QWORD *)(v12 + 8);
            }
            if ( v14 != v12 + 24 )
              _libc_free(v14);
          }
          while ( v11 != (int *)v12 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v16 = v50 + 56 * v4;
      if ( v16 != v50 )
      {
        do
        {
          v17 = *(unsigned int *)(v16 - 40);
          v18 = *(_QWORD *)(v16 - 48);
          v16 -= 56LL;
          v19 = (unsigned __int64 *)(v18 + 32 * v17);
          if ( (unsigned __int64 *)v18 != v19 )
          {
            do
            {
              v19 -= 4;
              if ( (unsigned __int64 *)*v19 != v19 + 2 )
                j_j___libc_free_0(*v19);
            }
            while ( (unsigned __int64 *)v18 != v19 );
            v18 = *(_QWORD *)(v16 + 8);
          }
          if ( v18 != v16 + 24 )
            _libc_free(v18);
        }
        while ( v16 != v50 );
        v48 = *(_QWORD *)a1;
      }
      if ( v48 != a1 + 16 )
        _libc_free(v48);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v47;
    }
  }
}
