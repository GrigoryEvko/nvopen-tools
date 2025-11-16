// Function: sub_2A8C4C0
// Address: 0x2a8c4c0
//
void __fastcall sub_2A8C4C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r14
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *i; // rcx
  __int64 v14; // r12
  __int64 v15; // rbx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  _QWORD *v22; // r12
  __int64 v23; // rbx
  unsigned __int64 v24; // rdi
  __int64 v25; // rbx
  bool v26; // cc
  unsigned __int64 v27; // rdi
  __int64 v28; // rax
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // rdi
  int v31; // r14d
  unsigned __int64 v32; // rbx
  __int64 v33; // rsi
  unsigned __int64 v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // [rsp-60h] [rbp-60h]
  __int64 v37; // [rsp-58h] [rbp-58h]
  unsigned __int64 v38; // [rsp-58h] [rbp-58h]
  int v39; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int64 v40; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != a2 )
  {
    v7 = (_QWORD *)(a2 + 16);
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v39 = *(_DWORD *)(a2 + 8);
      if ( v11 <= v9 )
      {
        v19 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v32 = *(_QWORD *)a1;
          v36 = 24 * v11;
          v33 = a2 + 24 * v11 + 16;
          do
          {
            v26 = *(_DWORD *)(v32 + 16) <= 0x40u;
            *(_QWORD *)v32 = *v7;
            if ( !v26 )
            {
              v34 = *(_QWORD *)(v32 + 8);
              if ( v34 )
                j_j___libc_free_0_0(v34);
            }
            v35 = v7[1];
            v7 += 3;
            v32 += 24LL;
            *(_QWORD *)(v32 - 16) = v35;
            *(_DWORD *)(v32 - 8) = *((_DWORD *)v7 - 2);
            *((_DWORD *)v7 - 2) = 0;
          }
          while ( v7 != (_QWORD *)v33 );
          v19 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
          v10 += v36;
        }
        v20 = v19 + 24 * v9;
        while ( v10 != v20 )
        {
          v20 -= 24LL;
          if ( *(_DWORD *)(v20 + 16) > 0x40u )
          {
            v21 = *(_QWORD *)(v20 + 8);
            if ( v21 )
              j_j___libc_free_0_0(v21);
          }
        }
        *(_DWORD *)(a1 + 8) = v39;
        v22 = *(_QWORD **)a2;
        v23 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v23 )
        {
          do
          {
            v23 -= 24;
            if ( *(_DWORD *)(v23 + 16) > 0x40u )
            {
              v24 = *(_QWORD *)(v23 + 8);
              if ( v24 )
                j_j___libc_free_0_0(v24);
            }
          }
          while ( v22 != (_QWORD *)v23 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          v29 = v10 + 24 * v9;
          while ( v29 != v10 )
          {
            while ( 1 )
            {
              v29 -= 24LL;
              if ( *(_DWORD *)(v29 + 16) <= 0x40u )
                break;
              v30 = *(_QWORD *)(v29 + 8);
              if ( !v30 )
                break;
              v38 = v11;
              j_j___libc_free_0_0(v30);
              v11 = v38;
              if ( v29 == v10 )
                goto LABEL_49;
            }
          }
LABEL_49:
          *(_DWORD *)(a1 + 8) = 0;
          v10 = sub_C8D7D0(a1, a1 + 16, v11, 0x18u, &v40, a6);
          sub_2A8AC40((__int64 *)a1, v10);
          v31 = v40;
          if ( a1 + 16 != *(_QWORD *)a1 )
            _libc_free(*(_QWORD *)a1);
          *(_QWORD *)a1 = v10;
          *(_DWORD *)(a1 + 12) = v31;
          v7 = *(_QWORD **)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v12 = *(_QWORD *)a2;
        }
        else
        {
          v12 = a2 + 16;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v37 = 24 * v9;
            v25 = a2 + 24 * v9 + 16;
            do
            {
              v26 = *(_DWORD *)(v10 + 16) <= 0x40u;
              *(_QWORD *)v10 = *v7;
              if ( !v26 )
              {
                v27 = *(_QWORD *)(v10 + 8);
                if ( v27 )
                  j_j___libc_free_0_0(v27);
              }
              v28 = v7[1];
              v7 += 3;
              v10 += 24LL;
              *(_QWORD *)(v10 - 16) = v28;
              *(_DWORD *)(v10 - 8) = *((_DWORD *)v7 - 2);
              *((_DWORD *)v7 - 2) = 0;
            }
            while ( v7 != (_QWORD *)v25 );
            v7 = *(_QWORD **)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v10 = v37 + *(_QWORD *)a1;
            v12 = *(_QWORD *)a2 + v37;
          }
        }
        for ( i = &v7[3 * v11]; i != (_QWORD *)v12; v10 += 24LL )
        {
          if ( v10 )
          {
            *(_QWORD *)v10 = *(_QWORD *)v12;
            *(_DWORD *)(v10 + 16) = *(_DWORD *)(v12 + 16);
            *(_QWORD *)(v10 + 8) = *(_QWORD *)(v12 + 8);
            *(_DWORD *)(v12 + 16) = 0;
          }
          v12 += 24;
        }
        *(_DWORD *)(a1 + 8) = v39;
        v14 = *(_QWORD *)a2;
        v15 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v15 )
        {
          do
          {
            v15 -= 24;
            if ( *(_DWORD *)(v15 + 16) > 0x40u )
            {
              v16 = *(_QWORD *)(v15 + 8);
              if ( v16 )
                j_j___libc_free_0_0(v16);
            }
          }
          while ( v14 != v15 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v17 = v10 + 24 * v9;
      if ( v17 != v10 )
      {
        do
        {
          v17 -= 24LL;
          if ( *(_DWORD *)(v17 + 16) > 0x40u )
          {
            v18 = *(_QWORD *)(v17 + 8);
            if ( v18 )
              j_j___libc_free_0_0(v18);
          }
        }
        while ( v17 != v10 );
        v10 = *(_QWORD *)a1;
      }
      if ( v10 != a1 + 16 )
        _libc_free(v10);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v7;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
