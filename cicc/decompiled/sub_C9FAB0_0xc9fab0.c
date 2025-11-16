// Function: sub_C9FAB0
// Address: 0xc9fab0
//
__int64 __fastcall sub_C9FAB0(__int64 a1, __int64 a2)
{
  int v3; // ecx
  __int64 *v4; // rdi
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 8);
  v4 = *(__int64 **)a1;
  if ( v3 )
  {
    if ( *v4 != -8 && *v4 )
    {
      v7 = v4;
    }
    else
    {
      v5 = v4 + 1;
      do
      {
        do
        {
          v6 = *v5;
          v7 = v5++;
        }
        while ( v6 == -8 );
      }
      while ( !v6 );
    }
    v8 = &v4[v3];
    if ( v8 != v7 )
    {
      do
      {
        while ( 1 )
        {
          v9 = *(_QWORD *)(*v7 + 8);
          if ( v9 )
          {
            sub_C9F930(*(_QWORD *)(*v7 + 8));
            a2 = 112;
            j_j___libc_free_0(v9, 112);
          }
          v10 = v7[1];
          v11 = v7 + 1;
          if ( !v10 || v10 == -8 )
            break;
          ++v7;
          if ( v11 == v8 )
            goto LABEL_16;
        }
        v12 = v7 + 2;
        do
        {
          do
          {
            v13 = *v12;
            v7 = v12++;
          }
          while ( v13 == -8 );
        }
        while ( !v13 );
      }
      while ( v7 != v8 );
LABEL_16:
      v4 = *(__int64 **)a1;
    }
  }
  if ( *(_DWORD *)(a1 + 12) )
  {
    v14 = *(unsigned int *)(a1 + 8);
    if ( (_DWORD)v14 )
    {
      v15 = 0;
      v26 = 8 * v14;
      do
      {
        v16 = v4[v15 / 8];
        if ( v16 != -8 && v16 )
        {
          v17 = *(_QWORD *)(v16 + 16);
          v23 = *(_QWORD *)v16 + 41LL;
          if ( *(_DWORD *)(v16 + 28) )
          {
            v18 = *(unsigned int *)(v16 + 24);
            if ( (_DWORD)v18 )
            {
              v19 = 8 * v18;
              v20 = 0;
              do
              {
                v21 = *(_QWORD *)(v17 + v20);
                if ( v21 && v21 != -8 )
                {
                  v24 = v19;
                  v25 = *(_QWORD *)v21 + 185LL;
                  sub_C9F8C0((const __m128i *)(v21 + 8));
                  a2 = v25;
                  sub_C7D6A0(v21, v25, 8);
                  v17 = *(_QWORD *)(v16 + 16);
                  v19 = v24;
                }
                v20 += 8;
              }
              while ( v19 != v20 );
            }
          }
          _libc_free(v17, a2);
          a2 = v23;
          sub_C7D6A0(v16, v23, 8);
          v4 = *(__int64 **)a1;
        }
        v15 += 8LL;
      }
      while ( v26 != v15 );
    }
  }
  return _libc_free(v4, a2);
}
