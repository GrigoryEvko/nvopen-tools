// Function: sub_B3DB20
// Address: 0xb3db20
//
void __fastcall sub_B3DB20(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v4; // r12
  unsigned __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  _DWORD *i; // r12
  int v9; // eax
  __int64 *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // rbx
  _QWORD *v15; // r15
  _QWORD *v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rax
  _QWORD *v19; // r14
  _QWORD *v20; // r15
  __int64 v21; // r15
  __int64 v22; // r12
  __int64 v23; // r14
  int v24; // eax
  __int64 v25; // rdi
  __int64 v26; // rbx
  __int64 v27; // r15
  unsigned __int64 v28; // r12
  int v29; // edx
  __int64 *v30; // rsi
  __int64 v31; // rdi
  unsigned int v32; // [rsp+Ch] [rbp-44h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  __int64 v35; // [rsp+18h] [rbp-38h]
  unsigned __int64 v36; // [rsp+18h] [rbp-38h]

  v33 = a2;
  if ( a1 != a2 )
  {
    v2 = *(_QWORD *)a1;
    v32 = *(_DWORD *)(a2 + 8);
    v4 = *(_QWORD *)a1;
    v34 = v32;
    v5 = *(unsigned int *)(a1 + 8);
    if ( v32 > v5 )
    {
      if ( v32 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v17 = v2 + 56 * v5;
        while ( v2 != v17 )
        {
          while ( 1 )
          {
            v18 = *(unsigned int *)(v17 - 40);
            v19 = *(_QWORD **)(v17 - 48);
            v17 -= 56;
            v18 *= 32;
            v20 = (_QWORD *)((char *)v19 + v18);
            if ( v19 != (_QWORD *)((char *)v19 + v18) )
            {
              do
              {
                v20 -= 4;
                if ( (_QWORD *)*v20 != v20 + 2 )
                {
                  a2 = v20[2] + 1LL;
                  j_j___libc_free_0(*v20, a2);
                }
              }
              while ( v19 != v20 );
              v19 = *(_QWORD **)(v17 + 8);
            }
            if ( v19 == (_QWORD *)(v17 + 24) )
              break;
            _libc_free(v19, a2);
            if ( v2 == v17 )
              goto LABEL_33;
          }
        }
LABEL_33:
        *(_DWORD *)(a1 + 8) = 0;
        sub_B3C890(a1, v32);
        v2 = *(_QWORD *)a1;
        v34 = *(unsigned int *)(v33 + 8);
        v5 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v26 = v2 + 8;
        v5 *= 56LL;
        v27 = *(_QWORD *)a2 + 8LL;
        v28 = v27 + v5;
        do
        {
          v29 = *(_DWORD *)(v27 - 8);
          v30 = (__int64 *)v27;
          v31 = v26;
          v27 += 56;
          v36 = v5;
          v26 += 56;
          *(_DWORD *)(v26 - 64) = v29;
          sub_B3C2C0(v31, v30);
          v5 = v36;
        }
        while ( v28 != v27 );
        v2 = *(_QWORD *)a1;
        v34 = *(unsigned int *)(v33 + 8);
      }
      v6 = v5 + v2;
      v7 = *(_QWORD *)v33 + 56 * v34;
      for ( i = (_DWORD *)(v5 + *(_QWORD *)v33); (_DWORD *)v7 != i; i += 14 )
      {
        while ( 1 )
        {
          if ( v6 )
          {
            v9 = *i;
            *(_DWORD *)(v6 + 16) = 0;
            *(_DWORD *)(v6 + 20) = 1;
            *(_DWORD *)v6 = v9;
            *(_QWORD *)(v6 + 8) = v6 + 24;
            if ( i[4] )
              break;
          }
          i += 14;
          v6 += 56;
          if ( (_DWORD *)v7 == i )
            goto LABEL_11;
        }
        v10 = (__int64 *)(i + 2);
        v11 = v6 + 8;
        v6 += 56;
        sub_B3C2C0(v11, v10);
      }
      goto LABEL_11;
    }
    v12 = *(_QWORD *)a1;
    if ( v32 )
    {
      v21 = *(_QWORD *)a2 + 8LL;
      v22 = v2 + 8;
      v35 = 56LL * v32;
      v23 = v21 + v35;
      do
      {
        v24 = *(_DWORD *)(v21 - 8);
        a2 = v21;
        v25 = v22;
        v21 += 56;
        v22 += 56;
        *(_DWORD *)(v22 - 64) = v24;
        sub_B3C2C0(v25, (__int64 *)a2);
      }
      while ( v23 != v21 );
      v12 = *(_QWORD *)a1;
      v5 = *(unsigned int *)(a1 + 8);
      v4 = v2 + v35;
    }
    v13 = v12 + 56 * v5;
    if ( v13 == v4 )
    {
LABEL_11:
      *(_DWORD *)(a1 + 8) = v32;
      return;
    }
    do
    {
      v14 = *(unsigned int *)(v13 - 40);
      v15 = *(_QWORD **)(v13 - 48);
      v13 -= 56;
      v16 = &v15[4 * v14];
      if ( v15 != v16 )
      {
        do
        {
          v16 -= 4;
          if ( (_QWORD *)*v16 != v16 + 2 )
          {
            a2 = v16[2] + 1LL;
            j_j___libc_free_0(*v16, a2);
          }
        }
        while ( v15 != v16 );
        v15 = *(_QWORD **)(v13 + 8);
      }
      if ( v15 != (_QWORD *)(v13 + 24) )
        _libc_free(v15, a2);
    }
    while ( v4 != v13 );
    *(_DWORD *)(a1 + 8) = v32;
  }
}
