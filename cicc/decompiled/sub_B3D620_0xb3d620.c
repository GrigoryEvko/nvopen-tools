// Function: sub_B3D620
// Address: 0xb3d620
//
void __fastcall sub_B3D620(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  unsigned __int64 v4; // rax
  int *v5; // r15
  __int64 v6; // r12
  int *v7; // rbx
  int v8; // eax
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // rbx
  _QWORD *v13; // r15
  _QWORD *v14; // rbx
  __int64 v15; // rbx
  __int64 v16; // r15
  _QWORD *v17; // r12
  _QWORD *v18; // r15
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // r12
  _QWORD *v22; // r15
  _QWORD *v23; // r12
  int *v24; // r14
  __int64 v25; // r12
  __int64 v26; // rbx
  _QWORD *v27; // r15
  _QWORD *v28; // rbx
  __int64 v29; // r12
  __int64 v30; // r15
  unsigned __int64 v31; // rbx
  int v32; // ecx
  __int64 v33; // rdi
  __int64 v34; // r15
  __int64 v35; // rax
  _QWORD *v36; // rbx
  _QWORD *v37; // r12
  __int64 v38; // rbx
  __int64 v39; // r15
  int v40; // ecx
  __int64 v41; // rdi
  unsigned int v42; // [rsp-5Ch] [rbp-5Ch]
  __int64 v43; // [rsp-58h] [rbp-58h]
  __int64 v44; // [rsp-50h] [rbp-50h]
  __int64 v45; // [rsp-48h] [rbp-48h]
  __int64 v46; // [rsp-48h] [rbp-48h]
  __int64 v47; // [rsp-40h] [rbp-40h]
  unsigned __int64 v48; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v3 = a2;
    v4 = *(unsigned int *)(a1 + 8);
    v44 = a2 + 16;
    v47 = *(_QWORD *)a1;
    v45 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v42 = *(_DWORD *)(a2 + 8);
      v43 = v42;
      if ( v42 <= v4 )
      {
        v19 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v38 = a2 + 24;
          v39 = v47 + 8;
          do
          {
            v40 = *(_DWORD *)(v38 - 8);
            a2 = v38;
            v41 = v39;
            v38 += 56;
            v39 += 56;
            *(_DWORD *)(v39 - 64) = v40;
            sub_B3BE00(v41, a2);
          }
          while ( v47 + 8 + 56LL * v42 != v39 );
          v19 = *(_QWORD *)a1;
          v45 = v47 + 56LL * v42;
          v4 = *(unsigned int *)(a1 + 8);
        }
        v20 = v19 + 56 * v4;
        while ( v45 != v20 )
        {
          v21 = *(unsigned int *)(v20 - 40);
          v22 = *(_QWORD **)(v20 - 48);
          v20 -= 56;
          v23 = &v22[4 * v21];
          if ( v22 != v23 )
          {
            do
            {
              v23 -= 4;
              if ( (_QWORD *)*v23 != v23 + 2 )
              {
                a2 = v23[2] + 1LL;
                j_j___libc_free_0(*v23, a2);
              }
            }
            while ( v22 != v23 );
            v22 = *(_QWORD **)(v20 + 8);
          }
          if ( v22 != (_QWORD *)(v20 + 24) )
            _libc_free(v22, a2);
        }
        *(_DWORD *)(a1 + 8) = v42;
        v24 = *(int **)v3;
        v25 = *(_QWORD *)v3 + 56LL * *(unsigned int *)(v3 + 8);
        if ( *(_QWORD *)v3 != v25 )
        {
          do
          {
            v26 = *(unsigned int *)(v25 - 40);
            v27 = *(_QWORD **)(v25 - 48);
            v25 -= 56;
            v28 = &v27[4 * v26];
            if ( v27 != v28 )
            {
              do
              {
                v28 -= 4;
                if ( (_QWORD *)*v28 != v28 + 2 )
                {
                  a2 = v28[2] + 1LL;
                  j_j___libc_free_0(*v28, a2);
                }
              }
              while ( v27 != v28 );
              v27 = *(_QWORD **)(v25 + 8);
            }
            if ( v27 != (_QWORD *)(v25 + 24) )
              _libc_free(v27, a2);
          }
          while ( v24 != (int *)v25 );
        }
      }
      else
      {
        if ( v42 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v34 = v47 + 56 * v4;
          while ( v47 != v34 )
          {
            v35 = *(unsigned int *)(v34 - 40);
            v36 = *(_QWORD **)(v34 - 48);
            v34 -= 56;
            v35 *= 32;
            v37 = (_QWORD *)((char *)v36 + v35);
            if ( v36 != (_QWORD *)((char *)v36 + v35) )
            {
              do
              {
                v37 -= 4;
                if ( (_QWORD *)*v37 != v37 + 2 )
                {
                  a2 = v37[2] + 1LL;
                  j_j___libc_free_0(*v37, a2);
                }
              }
              while ( v36 != v37 );
              v36 = *(_QWORD **)(v34 + 8);
            }
            if ( v36 != (_QWORD *)(v34 + 24) )
              _libc_free(v36, a2);
          }
          *(_DWORD *)(a1 + 8) = 0;
          a2 = v42;
          sub_B3C890(a1, v42);
          v5 = *(int **)v3;
          v43 = *(unsigned int *)(v3 + 8);
          v4 = 0;
          v44 = *(_QWORD *)v3;
          v47 = *(_QWORD *)a1;
        }
        else
        {
          v5 = (int *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v29 = v47 + 8;
            v30 = a2 + 24;
            v46 = 56 * v4;
            v4 *= 56LL;
            v31 = v47 + 8 + v4;
            do
            {
              v32 = *(_DWORD *)(v30 - 8);
              a2 = v30;
              v33 = v29;
              v29 += 56;
              v48 = v4;
              v30 += 56;
              *(_DWORD *)(v29 - 64) = v32;
              sub_B3BE00(v33, a2);
              v4 = v48;
            }
            while ( v31 != v29 );
            v44 = *(_QWORD *)v3;
            v5 = (int *)(*(_QWORD *)v3 + v46);
            v43 = *(unsigned int *)(v3 + 8);
            v47 = *(_QWORD *)a1;
          }
        }
        v6 = v4 + v47;
        v7 = (int *)(v44 + 56 * v43);
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
            v6 += 56;
            if ( v7 == v5 )
              goto LABEL_12;
          }
          a2 = (__int64)(v5 + 2);
          v9 = v6 + 8;
          v5 += 14;
          v6 += 56;
          sub_B3BE00(v9, a2);
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v42;
        v10 = *(_QWORD *)v3;
        v11 = *(_QWORD *)v3 + 56LL * *(unsigned int *)(v3 + 8);
        if ( *(_QWORD *)v3 != v11 )
        {
          do
          {
            v12 = *(unsigned int *)(v11 - 40);
            v13 = *(_QWORD **)(v11 - 48);
            v11 -= 56;
            v14 = &v13[4 * v12];
            if ( v13 != v14 )
            {
              do
              {
                v14 -= 4;
                if ( (_QWORD *)*v14 != v14 + 2 )
                {
                  a2 = v14[2] + 1LL;
                  j_j___libc_free_0(*v14, a2);
                }
              }
              while ( v13 != v14 );
              v13 = *(_QWORD **)(v11 + 8);
            }
            if ( v13 != (_QWORD *)(v11 + 24) )
              _libc_free(v13, a2);
          }
          while ( v10 != v11 );
        }
      }
      *(_DWORD *)(v3 + 8) = 0;
    }
    else
    {
      v15 = v47 + 56 * v4;
      if ( v15 != v47 )
      {
        do
        {
          v16 = *(unsigned int *)(v15 - 40);
          v17 = *(_QWORD **)(v15 - 48);
          v15 -= 56;
          v18 = &v17[4 * v16];
          if ( v17 != v18 )
          {
            do
            {
              v18 -= 4;
              if ( (_QWORD *)*v18 != v18 + 2 )
              {
                a2 = v18[2] + 1LL;
                j_j___libc_free_0(*v18, a2);
              }
            }
            while ( v17 != v18 );
            v17 = *(_QWORD **)(v15 + 8);
          }
          if ( v17 != (_QWORD *)(v15 + 24) )
            _libc_free(v17, a2);
        }
        while ( v15 != v47 );
        v45 = *(_QWORD *)a1;
      }
      if ( v45 != a1 + 16 )
        _libc_free(v45, a2);
      *(_QWORD *)a1 = *(_QWORD *)v3;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(v3 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(v3 + 12);
      *(_QWORD *)(v3 + 8) = 0;
      *(_QWORD *)v3 = v44;
    }
  }
}
