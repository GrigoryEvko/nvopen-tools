// Function: sub_273D6F0
// Address: 0x273d6f0
//
void __fastcall sub_273D6F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  _QWORD *v8; // r13
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rsi
  int v13; // r12d
  unsigned __int64 v14; // rdx
  _QWORD *v15; // rdx
  unsigned __int64 v16; // rax
  _QWORD *i; // rdi
  _QWORD *v18; // r12
  __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rdi
  _QWORD *v26; // r12
  __int64 v27; // rbx
  unsigned __int64 v28; // rdi
  __int64 v29; // rbx
  bool v30; // cc
  unsigned __int64 v31; // rdi
  __int64 v32; // rdx
  unsigned __int64 v33; // r13
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // r15
  __int64 v36; // rsi
  unsigned __int64 v37; // rdi
  __int64 v38; // rdx
  unsigned __int64 v39; // [rsp-50h] [rbp-50h]
  __int64 v40; // [rsp-50h] [rbp-50h]
  __int64 v41; // [rsp-48h] [rbp-48h]
  __int64 v42; // [rsp-48h] [rbp-48h]
  __int64 v43; // [rsp-48h] [rbp-48h]
  __int64 v44; // [rsp-40h] [rbp-40h]
  __int64 v45; // [rsp-40h] [rbp-40h]
  __int64 v46; // [rsp-40h] [rbp-40h]
  unsigned __int64 v47; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v48; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = a1;
    v8 = (_QWORD *)(a2 + 16);
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v12 = *(unsigned int *)(a2 + 8);
      v13 = v12;
      if ( v12 <= v10 )
      {
        v23 = *(_QWORD *)a1;
        if ( v12 )
        {
          v35 = *(_QWORD *)a1;
          v40 = 24 * v12;
          v36 = a2 + 24 * v12 + 16;
          do
          {
            v30 = *(_DWORD *)(v35 + 16) <= 0x40u;
            *(_QWORD *)v35 = *v8;
            if ( !v30 )
            {
              v37 = *(_QWORD *)(v35 + 8);
              if ( v37 )
              {
                v43 = v6;
                j_j___libc_free_0_0(v37);
                v6 = v43;
              }
            }
            v38 = v8[1];
            v8 += 3;
            v35 += 24LL;
            *(_QWORD *)(v35 - 16) = v38;
            *(_DWORD *)(v35 - 8) = *((_DWORD *)v8 - 2);
            *((_DWORD *)v8 - 2) = 0;
          }
          while ( v8 != (_QWORD *)v36 );
          v23 = *(_QWORD *)v6;
          v10 = *(unsigned int *)(v6 + 8);
          v11 = v9 + v40;
        }
        v24 = v23 + 24 * v10;
        while ( v11 != v24 )
        {
          v24 -= 24LL;
          if ( *(_DWORD *)(v24 + 16) > 0x40u )
          {
            v25 = *(_QWORD *)(v24 + 8);
            if ( v25 )
            {
              v46 = v6;
              j_j___libc_free_0_0(v25);
              v6 = v46;
            }
          }
        }
        *(_DWORD *)(v6 + 8) = v13;
        v26 = *(_QWORD **)a2;
        v27 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v27 )
        {
          do
          {
            v27 -= 24;
            if ( *(_DWORD *)(v27 + 16) > 0x40u )
            {
              v28 = *(_QWORD *)(v27 + 8);
              if ( v28 )
                j_j___libc_free_0_0(v28);
            }
          }
          while ( v26 != (_QWORD *)v27 );
        }
      }
      else
      {
        v14 = *(unsigned int *)(a1 + 12);
        if ( v12 > v14 )
        {
          v33 = v9 + 24 * v10;
          while ( v9 != v33 )
          {
            while ( 1 )
            {
              v33 -= 24LL;
              if ( *(_DWORD *)(v33 + 16) <= 0x40u )
                break;
              v34 = *(_QWORD *)(v33 + 8);
              if ( !v34 )
                break;
              v42 = v6;
              j_j___libc_free_0_0(v34);
              v6 = v42;
              if ( v9 == v33 )
                goto LABEL_49;
            }
          }
LABEL_49:
          *(_DWORD *)(v6 + 8) = 0;
          v48 = (unsigned __int64 *)v6;
          sub_273D600(v6, v12, v14, v6, a5, a6);
          v8 = *(_QWORD **)a2;
          v6 = (__int64)v48;
          v10 = 0;
          v12 = *(unsigned int *)(a2 + 8);
          v9 = *v48;
          v15 = *(_QWORD **)a2;
        }
        else
        {
          v15 = v8;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v10 *= 24LL;
            v39 = v10;
            v29 = a2 + v10 + 16;
            do
            {
              v30 = *(_DWORD *)(v11 + 16) <= 0x40u;
              *(_QWORD *)v11 = *v8;
              if ( !v30 )
              {
                v31 = *(_QWORD *)(v11 + 8);
                if ( v31 )
                {
                  v41 = v6;
                  v47 = v10;
                  j_j___libc_free_0_0(v31);
                  v6 = v41;
                  v10 = v47;
                }
              }
              v32 = v8[1];
              v8 += 3;
              v11 += 24LL;
              *(_QWORD *)(v11 - 16) = v32;
              *(_DWORD *)(v11 - 8) = *((_DWORD *)v8 - 2);
              *((_DWORD *)v8 - 2) = 0;
            }
            while ( v8 != (_QWORD *)v29 );
            v8 = *(_QWORD **)a2;
            v12 = *(unsigned int *)(a2 + 8);
            v9 = *(_QWORD *)v6;
            v15 = (_QWORD *)(*(_QWORD *)a2 + v39);
          }
        }
        v16 = v9 + v10;
        for ( i = &v8[3 * v12]; i != v15; v16 += 24LL )
        {
          if ( v16 )
          {
            *(_QWORD *)v16 = *v15;
            *(_DWORD *)(v16 + 16) = *((_DWORD *)v15 + 4);
            *(_QWORD *)(v16 + 8) = v15[1];
            *((_DWORD *)v15 + 4) = 0;
          }
          v15 += 3;
        }
        *(_DWORD *)(v6 + 8) = v13;
        v18 = *(_QWORD **)a2;
        v19 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v19 )
        {
          do
          {
            v19 -= 24;
            if ( *(_DWORD *)(v19 + 16) > 0x40u )
            {
              v20 = *(_QWORD *)(v19 + 8);
              if ( v20 )
                j_j___libc_free_0_0(v20);
            }
          }
          while ( v18 != (_QWORD *)v19 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v21 = v9 + 24 * v10;
      if ( v21 != v9 )
      {
        do
        {
          v21 -= 24LL;
          if ( *(_DWORD *)(v21 + 16) > 0x40u )
          {
            v22 = *(_QWORD *)(v21 + 8);
            if ( v22 )
            {
              v44 = v6;
              j_j___libc_free_0_0(v22);
              v6 = v44;
            }
          }
        }
        while ( v21 != v9 );
        v11 = *(_QWORD *)v6;
      }
      if ( v11 != v6 + 16 )
      {
        v45 = v6;
        _libc_free(v11);
        v6 = v45;
      }
      *(_QWORD *)v6 = *(_QWORD *)a2;
      *(_DWORD *)(v6 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(v6 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v8;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
