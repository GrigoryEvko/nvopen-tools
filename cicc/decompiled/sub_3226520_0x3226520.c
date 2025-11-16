// Function: sub_3226520
// Address: 0x3226520
//
void __fastcall sub_3226520(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 i; // rbx
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rsi
  __int64 j; // r8
  __int64 v17; // r13
  __int64 v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 v23; // rax
  unsigned __int64 v24; // r14
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-50h]
  unsigned int v29; // [rsp+14h] [rbp-3Ch]
  unsigned int v30; // [rsp+14h] [rbp-3Ch]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v6 = 112LL * *((unsigned int *)a1 + 2);
  v7 = *a1 + v6;
  if ( *a1 != v7 )
  {
    for ( i = *a1 + 32; ; i += 112 )
    {
      if ( a2 )
      {
        v10 = a2 + 32;
        *(_QWORD *)a2 = *(_QWORD *)(i - 32);
        v11 = *(_QWORD *)(i - 24);
        *(_QWORD *)(a2 + 16) = a2 + 32;
        *(_QWORD *)(a2 + 8) = v11;
        *(_DWORD *)(a2 + 24) = 0;
        *(_DWORD *)(a2 + 28) = 1;
        v12 = *(unsigned int *)(i - 8);
        if ( (_DWORD)v12 )
        {
          if ( a2 + 16 != i - 16 )
          {
            v13 = *(_QWORD *)(i - 16);
            v31 = v13;
            if ( i == v13 )
            {
              v14 = i;
              v15 = 1;
              if ( (_DWORD)v12 != 1 )
              {
                v29 = *(_DWORD *)(i - 8);
                sub_32262D0(a2 + 16, (unsigned int)v12, v6, v12, (unsigned int)v12, a6);
                v10 = *(_QWORD *)(a2 + 16);
                v14 = *(_QWORD *)(i - 16);
                v15 = *(unsigned int *)(i - 8);
                v12 = v29;
              }
              for ( j = v14 + 80 * v15; j != v14; v10 += 80 )
              {
                if ( v10 )
                {
                  v20 = *(_QWORD *)v14;
                  *(_DWORD *)(v10 + 16) = 0;
                  *(_DWORD *)(v10 + 20) = 2;
                  *(_QWORD *)v10 = v20;
                  *(_QWORD *)(v10 + 8) = v10 + 24;
                  if ( *(_DWORD *)(v14 + 16) )
                  {
                    v27 = j;
                    v30 = v12;
                    sub_3218940(v10 + 8, (char **)(v14 + 8), v6, v12, j, a6);
                    j = v27;
                    v12 = v30;
                  }
                  *(_BYTE *)(v10 + 72) = *(_BYTE *)(v14 + 72);
                }
                v14 += 80;
              }
              *(_DWORD *)(a2 + 24) = v12;
              v17 = *(_QWORD *)(v31 - 16);
              v18 = v17 + 80LL * *(unsigned int *)(v31 - 8);
              while ( v17 != v18 )
              {
                v18 -= 80;
                v19 = *(_QWORD *)(v18 + 8);
                if ( v19 != v18 + 24 )
                  _libc_free(v19);
              }
              *(_DWORD *)(v31 - 8) = 0;
            }
            else
            {
              *(_QWORD *)(a2 + 16) = v13;
              *(_DWORD *)(a2 + 24) = *(_DWORD *)(i - 8);
              *(_DWORD *)(a2 + 28) = *(_DWORD *)(i - 4);
              *(_QWORD *)(i - 16) = i;
              *(_DWORD *)(i - 4) = 0;
              *(_DWORD *)(i - 8) = 0;
            }
          }
        }
      }
      a2 += 112;
      if ( v7 == i + 80 )
        break;
    }
    v21 = *a1;
    v22 = *a1 + 112LL * *((unsigned int *)a1 + 2);
    while ( v22 != v21 )
    {
      v23 = *(unsigned int *)(v22 - 88);
      v24 = *(_QWORD *)(v22 - 96);
      v22 -= 112;
      v25 = v24 + 80 * v23;
      if ( v24 != v25 )
      {
        do
        {
          v25 -= 80LL;
          v26 = *(_QWORD *)(v25 + 8);
          if ( v26 != v25 + 24 )
            _libc_free(v26);
        }
        while ( v24 != v25 );
        v24 = *(_QWORD *)(v22 + 16);
      }
      if ( v24 != v22 + 32 )
        _libc_free(v24);
    }
  }
}
