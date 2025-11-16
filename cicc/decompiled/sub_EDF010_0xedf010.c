// Function: sub_EDF010
// Address: 0xedf010
//
void __fastcall sub_EDF010(__int64 a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 *v3; // rbx
  __int64 v4; // r13
  __int64 *v5; // rax
  __int64 *v6; // r14
  __int64 *i; // rcx
  __int64 *v8; // r13
  __int64 *v9; // r14
  __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // rbx
  _QWORD *v13; // r12
  __int64 *v14; // r15
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // r12
  _QWORD *v18; // r14
  __int64 *v19; // rax
  __int64 *v20; // r14
  __int64 v21; // rdi
  __int64 v22; // r15
  __int64 v23; // rbx
  _QWORD *v24; // r13
  __int64 *v25; // r13
  __int64 *v26; // r14
  __int64 v27; // rdi
  __int64 v28; // r15
  __int64 v29; // rbx
  _QWORD *v30; // r12
  __int64 v31; // r14
  __int64 v32; // r13
  __int64 v33; // r15
  _QWORD *v34; // r12
  __int64 *v35; // r12
  __int64 v36; // rdi
  __int64 v37; // r14
  __int64 v38; // rbx
  _QWORD *v39; // r13
  __int64 v40; // r14
  __int64 v41; // r13
  __int64 v42; // r15
  _QWORD *v43; // r8
  __int64 *v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+10h] [rbp-60h]
  __int64 *v46; // [rsp+10h] [rbp-60h]
  unsigned int v47; // [rsp+1Ch] [rbp-54h]
  unsigned __int64 v48; // [rsp+20h] [rbp-50h]
  __int64 v49; // [rsp+20h] [rbp-50h]
  __int64 v51; // [rsp+30h] [rbp-40h]
  __int64 *v52; // [rsp+38h] [rbp-38h]
  __int64 *v53; // [rsp+38h] [rbp-38h]
  _QWORD *v54; // [rsp+38h] [rbp-38h]

  v51 = a2;
  if ( a1 != a2 )
  {
    v2 = *(__int64 **)a1;
    v3 = (__int64 *)(a2 + 16);
    v48 = *(unsigned int *)(a1 + 8);
    v52 = *(__int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v47 = *(_DWORD *)(a2 + 8);
      v4 = v47;
      if ( v47 > v48 )
      {
        if ( v47 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v35 = &v52[3 * v48];
          if ( v35 != v52 )
          {
            do
            {
              v36 = *(v35 - 3);
              v37 = *(v35 - 2);
              v35 -= 3;
              v38 = v36;
              if ( v37 != v36 )
              {
                do
                {
                  v39 = *(_QWORD **)(v38 + 8);
                  if ( v39 )
                  {
                    if ( (_QWORD *)*v39 != v39 + 2 )
                      j_j___libc_free_0(*v39, v39[2] + 1LL);
                    j_j___libc_free_0(v39, 32);
                  }
                  v38 += 32;
                }
                while ( v37 != v38 );
                v36 = *v35;
              }
              if ( v36 )
                j_j___libc_free_0(v36, v35[2] - v36);
            }
            while ( v35 != v52 );
            v4 = v47;
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_C170B0(a1, v4);
          v48 = 0;
          v3 = *(__int64 **)a2;
          v4 = *(unsigned int *)(a2 + 8);
          v52 = *(__int64 **)a1;
          v5 = *(__int64 **)a2;
        }
        else
        {
          v5 = (__int64 *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v48 *= 24LL;
            v44 = (__int64 *)((char *)v2 + v48);
            do
            {
              v31 = *v2;
              v32 = v2[1];
              v45 = v2[2];
              v33 = v31;
              *v2 = *v3;
              v2[1] = v3[1];
              v2[2] = v3[2];
              *v3 = 0;
              v3[1] = 0;
              v3[2] = 0;
              if ( v31 != v32 )
              {
                v53 = v2;
                do
                {
                  v34 = *(_QWORD **)(v33 + 8);
                  if ( v34 )
                  {
                    if ( (_QWORD *)*v34 != v34 + 2 )
                      j_j___libc_free_0(*v34, v34[2] + 1LL);
                    j_j___libc_free_0(v34, 32);
                  }
                  v33 += 32;
                }
                while ( v32 != v33 );
                v2 = v53;
              }
              if ( v31 )
                j_j___libc_free_0(v31, v45 - v31);
              v3 += 3;
              v2 += 3;
            }
            while ( v2 != v44 );
            v3 = *(__int64 **)a2;
            v4 = *(unsigned int *)(a2 + 8);
            v52 = *(__int64 **)a1;
            v5 = (__int64 *)(*(_QWORD *)a2 + v48);
          }
        }
        v6 = (__int64 *)((char *)v52 + v48);
        for ( i = &v3[3 * v4]; i != v5; v6 += 3 )
        {
          if ( v6 )
          {
            *v6 = *v5;
            v6[1] = v5[1];
            v6[2] = v5[2];
            v5[2] = 0;
            v5[1] = 0;
            *v5 = 0;
          }
          v5 += 3;
        }
        *(_DWORD *)(a1 + 8) = v47;
        v8 = *(__int64 **)a2;
        v9 = (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
        if ( *(__int64 **)a2 != v9 )
        {
          do
          {
            v10 = *(v9 - 3);
            v11 = *(v9 - 2);
            v9 -= 3;
            v12 = v10;
            if ( v11 != v10 )
            {
              do
              {
                v13 = *(_QWORD **)(v12 + 8);
                if ( v13 )
                {
                  if ( (_QWORD *)*v13 != v13 + 2 )
                    j_j___libc_free_0(*v13, v13[2] + 1LL);
                  j_j___libc_free_0(v13, 32);
                }
                v12 += 32;
              }
              while ( v11 != v12 );
              v10 = *v9;
            }
            if ( v10 )
              j_j___libc_free_0(v10, v9[2] - v10);
          }
          while ( v8 != v9 );
        }
LABEL_21:
        *(_DWORD *)(a2 + 8) = 0;
        return;
      }
      v19 = *(__int64 **)a1;
      if ( v47 )
      {
        v46 = &v2[3 * v47];
        do
        {
          v40 = *v2;
          v41 = v2[1];
          v49 = v2[2];
          v42 = v40;
          *v2 = *v3;
          v2[1] = v3[1];
          v2[2] = v3[2];
          *v3 = 0;
          v3[1] = 0;
          for ( v3[2] = 0; v41 != v42; v42 += 32 )
          {
            v43 = *(_QWORD **)(v42 + 8);
            if ( v43 )
            {
              if ( (_QWORD *)*v43 != v43 + 2 )
              {
                v54 = *(_QWORD **)(v42 + 8);
                j_j___libc_free_0(*v43, v43[2] + 1LL);
                v43 = v54;
              }
              j_j___libc_free_0(v43, 32);
            }
          }
          if ( v40 )
            j_j___libc_free_0(v40, v49 - v40);
          v3 += 3;
          v2 += 3;
        }
        while ( v2 != v46 );
        v19 = *(__int64 **)a1;
        v48 = *(unsigned int *)(a1 + 8);
      }
      v20 = &v19[3 * v48];
      while ( v2 != v20 )
      {
        v21 = *(v20 - 3);
        v22 = *(v20 - 2);
        v20 -= 3;
        v23 = v21;
        if ( v22 != v21 )
        {
          do
          {
            v24 = *(_QWORD **)(v23 + 8);
            if ( v24 )
            {
              if ( (_QWORD *)*v24 != v24 + 2 )
                j_j___libc_free_0(*v24, v24[2] + 1LL);
              j_j___libc_free_0(v24, 32);
            }
            v23 += 32;
          }
          while ( v22 != v23 );
          v21 = *v20;
        }
        if ( v21 )
          j_j___libc_free_0(v21, v20[2] - v21);
      }
      *(_DWORD *)(a1 + 8) = v47;
      v25 = *(__int64 **)a2;
      v26 = (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
      if ( *(__int64 **)a2 == v26 )
        goto LABEL_21;
      do
      {
        v27 = *(v26 - 3);
        v28 = *(v26 - 2);
        v26 -= 3;
        v29 = v27;
        if ( v28 != v27 )
        {
          do
          {
            v30 = *(_QWORD **)(v29 + 8);
            if ( v30 )
            {
              if ( (_QWORD *)*v30 != v30 + 2 )
                j_j___libc_free_0(*v30, v30[2] + 1LL);
              j_j___libc_free_0(v30, 32);
            }
            v29 += 32;
          }
          while ( v28 != v29 );
          v27 = *v26;
        }
        if ( v27 )
          j_j___libc_free_0(v27, v26[2] - v27);
      }
      while ( v25 != v26 );
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v14 = &v2[3 * *(unsigned int *)(a1 + 8)];
      if ( v14 != v2 )
      {
        do
        {
          v15 = *(v14 - 3);
          v16 = *(v14 - 2);
          v14 -= 3;
          v17 = v15;
          if ( v16 != v15 )
          {
            do
            {
              v18 = *(_QWORD **)(v17 + 8);
              if ( v18 )
              {
                if ( (_QWORD *)*v18 != v18 + 2 )
                  j_j___libc_free_0(*v18, v18[2] + 1LL);
                a2 = 32;
                j_j___libc_free_0(v18, 32);
              }
              v17 += 32;
            }
            while ( v16 != v17 );
            v15 = *v14;
          }
          if ( v15 )
          {
            a2 = v14[2] - v15;
            j_j___libc_free_0(v15, a2);
          }
        }
        while ( v14 != v52 );
        v2 = *(__int64 **)a1;
      }
      if ( v2 != (__int64 *)(a1 + 16) )
        _libc_free(v2, a2);
      *(_QWORD *)a1 = *(_QWORD *)v51;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(v51 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(v51 + 12);
      *(_QWORD *)v51 = v3;
      *(_QWORD *)(v51 + 8) = 0;
    }
  }
}
