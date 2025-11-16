// Function: sub_205B3F0
// Address: 0x205b3f0
//
void __fastcall sub_205B3F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int64 v12; // r14
  _QWORD *v13; // r15
  __int64 v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // r14
  _DWORD *i; // r12
  int v19; // eax
  __int64 *v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // rcx
  __int64 v24; // r15
  unsigned __int64 v25; // r12
  __int64 v26; // rdx
  __int64 *v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // r14
  __int64 v31; // rbx
  unsigned __int64 v32; // r15
  _QWORD *v33; // rbx
  __int64 v34; // r15
  __int64 v35; // rcx
  __int64 v36; // r12
  __int64 v37; // r14
  int v38; // eax
  __int64 *v39; // rsi
  __int64 v40; // rdi
  unsigned int v41; // [rsp+Ch] [rbp-44h]
  __int64 v43; // [rsp+18h] [rbp-38h]
  unsigned __int64 v44; // [rsp+18h] [rbp-38h]
  __int64 v45; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 )
  {
    v6 = *(_QWORD *)a1;
    v41 = *(_DWORD *)(a2 + 8);
    v8 = *(_QWORD *)a1;
    v43 = v41;
    v9 = *(unsigned int *)(a1 + 8);
    if ( v41 > v9 )
    {
      if ( v41 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        if ( *(_DWORD *)(a1 + 8) )
        {
          v22 = v6 + 8;
          v23 = *(_QWORD *)a2;
          v9 *= 56LL;
          v24 = *(_QWORD *)a2 + 8LL;
          v25 = v24 + v9;
          do
          {
            v26 = *(unsigned int *)(v24 - 8);
            v27 = (__int64 *)v24;
            v28 = v22;
            v24 += 56;
            v44 = v9;
            v22 += 56;
            *(_DWORD *)(v22 - 64) = v26;
            sub_2055490(v28, v27, v26, v23, a5, a6);
            v9 = v44;
          }
          while ( v25 != v24 );
          v6 = *(_QWORD *)a1;
          v43 = *(unsigned int *)(a2 + 8);
        }
      }
      else
      {
        v10 = v6 + 56 * v9;
        while ( v10 != v6 )
        {
          while ( 1 )
          {
            v11 = *(unsigned int *)(v10 - 40);
            v12 = *(_QWORD *)(v10 - 48);
            v10 -= 56;
            v11 *= 32;
            v13 = (_QWORD *)(v12 + v11);
            if ( v12 != v12 + v11 )
            {
              do
              {
                v13 -= 4;
                if ( (_QWORD *)*v13 != v13 + 2 )
                  j_j___libc_free_0(*v13, v13[2] + 1LL);
              }
              while ( (_QWORD *)v12 != v13 );
              v12 = *(_QWORD *)(v10 + 8);
            }
            if ( v12 == v10 + 24 )
              break;
            _libc_free(v12);
            if ( v10 == v6 )
              goto LABEL_13;
          }
        }
LABEL_13:
        *(_DWORD *)(a1 + 8) = 0;
        sub_15EB820(a1, v41);
        v6 = *(_QWORD *)a1;
        v43 = *(unsigned int *)(a2 + 8);
        v9 = 0;
      }
      v14 = v9 + v6;
      v15 = v43;
      v16 = 7 * v43;
      v17 = *(_QWORD *)a2 + 56 * v43;
      for ( i = (_DWORD *)(v9 + *(_QWORD *)a2); (_DWORD *)v17 != i; i += 14 )
      {
        while ( 1 )
        {
          if ( v14 )
          {
            v19 = *i;
            *(_DWORD *)(v14 + 16) = 0;
            *(_DWORD *)(v14 + 20) = 1;
            *(_DWORD *)v14 = v19;
            *(_QWORD *)(v14 + 8) = v14 + 24;
            if ( i[4] )
              break;
          }
          i += 14;
          v14 += 56;
          if ( (_DWORD *)v17 == i )
            goto LABEL_20;
        }
        v20 = (__int64 *)(i + 2);
        v21 = v14 + 8;
        v14 += 56;
        sub_2055490(v21, v20, v16, v15, a5, a6);
      }
      goto LABEL_20;
    }
    v29 = *(_QWORD *)a1;
    if ( v41 )
    {
      v34 = *(_QWORD *)a2 + 8LL;
      v35 = 56LL * v41;
      v36 = v6 + 8;
      v45 = v35;
      v37 = v34 + v35;
      do
      {
        v38 = *(_DWORD *)(v34 - 8);
        v39 = (__int64 *)v34;
        v40 = v36;
        v34 += 56;
        v36 += 56;
        *(_DWORD *)(v36 - 64) = v38;
        sub_2055490(v40, v39, a3, v35, a5, a6);
      }
      while ( v37 != v34 );
      v29 = *(_QWORD *)a1;
      v9 = *(unsigned int *)(a1 + 8);
      v8 = v6 + v45;
    }
    v30 = v29 + 56 * v9;
    if ( v30 == v8 )
    {
LABEL_20:
      *(_DWORD *)(a1 + 8) = v41;
      return;
    }
    do
    {
      v31 = *(unsigned int *)(v30 - 40);
      v32 = *(_QWORD *)(v30 - 48);
      v30 -= 56;
      v33 = (_QWORD *)(v32 + 32 * v31);
      if ( (_QWORD *)v32 != v33 )
      {
        do
        {
          v33 -= 4;
          if ( (_QWORD *)*v33 != v33 + 2 )
            j_j___libc_free_0(*v33, v33[2] + 1LL);
        }
        while ( (_QWORD *)v32 != v33 );
        v32 = *(_QWORD *)(v30 + 8);
      }
      if ( v32 != v30 + 24 )
        _libc_free(v32);
    }
    while ( v8 != v30 );
    *(_DWORD *)(a1 + 8) = v41;
  }
}
