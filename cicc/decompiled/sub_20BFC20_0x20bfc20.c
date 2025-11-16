// Function: sub_20BFC20
// Address: 0x20bfc20
//
void __fastcall sub_20BFC20(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rbx
  __int64 v8; // r15
  unsigned __int64 v9; // r14
  _QWORD *v10; // r15
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // rbx
  _QWORD *v15; // r15
  int *v16; // r15
  unsigned __int64 v17; // r12
  __int64 v18; // rcx
  int *v19; // rbx
  int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // r14
  __int64 v24; // r12
  __int64 v25; // rbx
  unsigned __int64 v26; // r15
  _QWORD *v27; // rbx
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // rbx
  __int64 v30; // r12
  unsigned __int64 v31; // r15
  _QWORD *v32; // r12
  _DWORD *v33; // r14
  __int64 v34; // r12
  __int64 v35; // rbx
  unsigned __int64 v36; // r15
  _QWORD *v37; // rbx
  __int64 v38; // rbx
  __int64 v39; // r15
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // rdi
  __int64 v43; // r12
  __int64 v44; // r14
  __int64 v45; // r15
  __int64 v46; // rbx
  __int64 v47; // rcx
  __int64 v48; // rsi
  __int64 v49; // rdi
  unsigned int v50; // [rsp+4h] [rbp-4Ch]
  unsigned __int64 v51; // [rsp+8h] [rbp-48h]
  __int64 v52; // [rsp+10h] [rbp-40h]
  unsigned __int64 v53; // [rsp+10h] [rbp-40h]

  if ( a1 != a2 )
  {
    v2 = *(_QWORD *)a1;
    v3 = a2 + 16;
    v4 = *(unsigned int *)(a1 + 8);
    v52 = a2 + 16;
    v6 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v50 = *(_DWORD *)(a2 + 8);
      v51 = v50;
      if ( v50 <= v4 )
      {
        v28 = *(_QWORD *)a1;
        if ( v50 )
        {
          v38 = a2 + 24;
          v39 = v2 + 8;
          do
          {
            v40 = *(unsigned int *)(v38 - 8);
            v41 = v38;
            v42 = v39;
            v38 += 56;
            v39 += 56;
            *(_DWORD *)(v39 - 64) = v40;
            sub_20BDAA0(v42, v41, v3, v40);
          }
          while ( v2 + 8 + 56LL * v50 != v39 );
          v6 = v2 + 56LL * v50;
          v28 = *(_QWORD *)a1;
          v4 = *(unsigned int *)(a1 + 8);
        }
        v29 = v28 + 56 * v4;
        while ( v6 != v29 )
        {
          v30 = *(unsigned int *)(v29 - 40);
          v31 = *(_QWORD *)(v29 - 48);
          v29 -= 56LL;
          v32 = (_QWORD *)(v31 + 32 * v30);
          if ( (_QWORD *)v31 != v32 )
          {
            do
            {
              v32 -= 4;
              if ( (_QWORD *)*v32 != v32 + 2 )
                j_j___libc_free_0(*v32, v32[2] + 1LL);
            }
            while ( (_QWORD *)v31 != v32 );
            v31 = *(_QWORD *)(v29 + 8);
          }
          if ( v31 != v29 + 24 )
            _libc_free(v31);
        }
        *(_DWORD *)(a1 + 8) = v50;
        v33 = *(_DWORD **)a2;
        v34 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v34 )
        {
          do
          {
            v35 = *(unsigned int *)(v34 - 40);
            v36 = *(_QWORD *)(v34 - 48);
            v34 -= 56;
            v37 = (_QWORD *)(v36 + 32 * v35);
            if ( (_QWORD *)v36 != v37 )
            {
              do
              {
                v37 -= 4;
                if ( (_QWORD *)*v37 != v37 + 2 )
                  j_j___libc_free_0(*v37, v37[2] + 1LL);
              }
              while ( (_QWORD *)v36 != v37 );
              v36 = *(_QWORD *)(v34 + 8);
            }
            if ( v36 != v34 + 24 )
              _libc_free(v36);
          }
          while ( v33 != (_DWORD *)v34 );
        }
      }
      else
      {
        v11 = *(unsigned int *)(a1 + 12);
        if ( v50 <= v11 )
        {
          v16 = (int *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v43 = v2 + 8;
            v44 = 56 * v4;
            v45 = a2 + 24;
            v4 = v44;
            v46 = v43 + v44;
            do
            {
              v47 = *(unsigned int *)(v45 - 8);
              v48 = v45;
              v49 = v43;
              v43 += 56;
              v53 = v4;
              v45 += 56;
              *(_DWORD *)(v43 - 64) = v47;
              sub_20BDAA0(v49, v48, v11, v47);
              v4 = v53;
            }
            while ( v46 != v43 );
            v11 = *(unsigned int *)(a2 + 8);
            v52 = *(_QWORD *)a2;
            v16 = (int *)(*(_QWORD *)a2 + v44);
            v51 = v11;
            v2 = *(_QWORD *)a1;
          }
        }
        else
        {
          v12 = v2 + 56 * v4;
          while ( v12 != v2 )
          {
            while ( 1 )
            {
              v13 = *(unsigned int *)(v12 - 40);
              v14 = *(_QWORD *)(v12 - 48);
              v12 -= 56LL;
              v13 *= 32;
              v15 = (_QWORD *)(v14 + v13);
              if ( v14 != v14 + v13 )
              {
                do
                {
                  v15 -= 4;
                  if ( (_QWORD *)*v15 != v15 + 2 )
                    j_j___libc_free_0(*v15, v15[2] + 1LL);
                }
                while ( (_QWORD *)v14 != v15 );
                v14 = *(_QWORD *)(v12 + 8);
              }
              if ( v14 == v12 + 24 )
                break;
              _libc_free(v14);
              if ( v12 == v2 )
                goto LABEL_28;
            }
          }
LABEL_28:
          *(_DWORD *)(a1 + 8) = 0;
          sub_15EB820(a1, v50);
          v16 = *(int **)a2;
          v2 = *(_QWORD *)a1;
          v51 = *(unsigned int *)(a2 + 8);
          v4 = 0;
          v52 = *(_QWORD *)a2;
        }
        v17 = v4 + v2;
        v18 = v52;
        v19 = (int *)(v52 + 56 * v51);
        while ( v19 != v16 )
        {
          while ( 1 )
          {
            if ( v17 )
            {
              v20 = *v16;
              *(_DWORD *)(v17 + 16) = 0;
              *(_DWORD *)(v17 + 20) = 1;
              *(_DWORD *)v17 = v20;
              *(_QWORD *)(v17 + 8) = v17 + 24;
              if ( v16[4] )
                break;
            }
            v16 += 14;
            v17 += 56LL;
            if ( v19 == v16 )
              goto LABEL_36;
          }
          v21 = (__int64)(v16 + 2);
          v22 = v17 + 8;
          v16 += 14;
          v17 += 56LL;
          sub_20BDAA0(v22, v21, v11, v18);
        }
LABEL_36:
        *(_DWORD *)(a1 + 8) = v50;
        v23 = *(_QWORD *)a2;
        v24 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v24 )
        {
          do
          {
            v25 = *(unsigned int *)(v24 - 40);
            v26 = *(_QWORD *)(v24 - 48);
            v24 -= 56;
            v27 = (_QWORD *)(v26 + 32 * v25);
            if ( (_QWORD *)v26 != v27 )
            {
              do
              {
                v27 -= 4;
                if ( (_QWORD *)*v27 != v27 + 2 )
                  j_j___libc_free_0(*v27, v27[2] + 1LL);
              }
              while ( (_QWORD *)v26 != v27 );
              v26 = *(_QWORD *)(v24 + 8);
            }
            if ( v26 != v24 + 24 )
              _libc_free(v26);
          }
          while ( v23 != v24 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v7 = v2 + 56 * v4;
      if ( v7 != v2 )
      {
        do
        {
          v8 = *(unsigned int *)(v7 - 40);
          v9 = *(_QWORD *)(v7 - 48);
          v7 -= 56LL;
          v10 = (_QWORD *)(v9 + 32 * v8);
          if ( (_QWORD *)v9 != v10 )
          {
            do
            {
              v10 -= 4;
              if ( (_QWORD *)*v10 != v10 + 2 )
                j_j___libc_free_0(*v10, v10[2] + 1LL);
            }
            while ( (_QWORD *)v9 != v10 );
            v9 = *(_QWORD *)(v7 + 8);
          }
          if ( v9 != v7 + 24 )
            _libc_free(v9);
        }
        while ( v7 != v2 );
        v6 = *(_QWORD *)a1;
      }
      if ( v6 != a1 + 16 )
        _libc_free(v6);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v52;
    }
  }
}
