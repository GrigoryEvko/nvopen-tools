// Function: sub_2B42980
// Address: 0x2b42980
//
void __fastcall sub_2B42980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  unsigned __int64 v9; // rbx
  __int64 v10; // r14
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rbx
  unsigned __int64 v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // rbx
  __int64 v20; // rax
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r13
  unsigned __int64 *v23; // rbx
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // r15
  __int64 v26; // rdi
  unsigned __int64 *v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  int v32; // r15d
  unsigned __int64 v33; // r15
  __int64 v34; // rdi
  int v35; // [rsp-54h] [rbp-54h]
  __int64 v36; // [rsp-50h] [rbp-50h]
  __int64 v37; // [rsp-50h] [rbp-50h]
  __int64 v38; // [rsp-50h] [rbp-50h]
  __int64 v39; // [rsp-50h] [rbp-50h]
  unsigned __int64 v40; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != a2 )
  {
    v6 = a2 + 16;
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v35 = *(_DWORD *)(a2 + 8);
      if ( v11 <= v9 )
      {
        v20 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v33 = v10 + (v11 << 6);
          do
          {
            v34 = v10;
            v39 = v6;
            v10 += 64;
            sub_2B0B060(v34, v6, v6, a4, a5, a6);
            v6 = v39 + 64;
          }
          while ( v10 != v33 );
          v20 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
        }
        v21 = (unsigned __int64 *)(v20 + (v9 << 6));
        while ( (unsigned __int64 *)v10 != v21 )
        {
          v21 -= 8;
          if ( (unsigned __int64 *)*v21 != v21 + 2 )
            _libc_free(*v21);
        }
        *(_DWORD *)(a1 + 8) = v35;
        v22 = *(unsigned __int64 **)a2;
        v23 = (unsigned __int64 *)(*(_QWORD *)a2 + ((unsigned __int64)*(unsigned int *)(a2 + 8) << 6));
        if ( *(unsigned __int64 **)a2 != v23 )
        {
          do
          {
            v23 -= 8;
            if ( (unsigned __int64 *)*v23 != v23 + 2 )
              _libc_free(*v23);
          }
          while ( v22 != v23 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          v27 = (unsigned __int64 *)(v10 + (v9 << 6));
          while ( (unsigned __int64 *)v10 != v27 )
          {
            while ( 1 )
            {
              v27 -= 8;
              if ( (unsigned __int64 *)*v27 == v27 + 2 )
                break;
              _libc_free(*v27);
              if ( (unsigned __int64 *)v10 == v27 )
                goto LABEL_43;
            }
          }
LABEL_43:
          *(_DWORD *)(a1 + 8) = 0;
          v10 = sub_C8D7D0(a1, a1 + 16, v11, 0x40u, &v40, a6);
          sub_2B428C0(a1, v10, v28, v29, v30, v31);
          v32 = v40;
          if ( a1 + 16 != *(_QWORD *)a1 )
            _libc_free(*(_QWORD *)a1);
          *(_QWORD *)a1 = v10;
          *(_DWORD *)(a1 + 12) = v32;
          v6 = *(_QWORD *)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v12 = *(_QWORD *)a2;
        }
        else
        {
          v12 = a2 + 16;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v24 = v9 << 6;
            v25 = v10 + v24;
            do
            {
              v26 = v10;
              v38 = v6;
              v10 += 64;
              sub_2B0B060(v26, v6, v6, a4, a5, a6);
              v6 = v38 + 64;
            }
            while ( v25 != v10 );
            v6 = *(_QWORD *)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v10 = v24 + *(_QWORD *)a1;
            v12 = *(_QWORD *)a2 + v24;
          }
        }
        v13 = v12;
        v14 = v6 + (v11 << 6);
        if ( v14 != v12 )
        {
          do
          {
            while ( 1 )
            {
              if ( v10 )
              {
                *(_DWORD *)(v10 + 8) = 0;
                *(_QWORD *)v10 = v10 + 16;
                *(_DWORD *)(v10 + 12) = 3;
                if ( *(_DWORD *)(v13 + 8) )
                  break;
              }
              v13 += 64;
              v10 += 64;
              if ( v14 == v13 )
                goto LABEL_12;
            }
            v15 = v13;
            v16 = v10;
            v13 += 64;
            v10 += 64;
            sub_2B0B060(v16, v15, v6, a4, a5, a6);
          }
          while ( v14 != v13 );
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v35;
        v17 = *(unsigned __int64 **)a2;
        v18 = (unsigned __int64 *)(*(_QWORD *)a2 + ((unsigned __int64)*(unsigned int *)(a2 + 8) << 6));
        if ( *(unsigned __int64 **)a2 != v18 )
        {
          do
          {
            v18 -= 8;
            if ( (unsigned __int64 *)*v18 != v18 + 2 )
              _libc_free(*v18);
          }
          while ( v17 != v18 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v19 = (unsigned __int64 *)(v10 + (v9 << 6));
      if ( v19 != (unsigned __int64 *)v10 )
      {
        do
        {
          v19 -= 8;
          if ( (unsigned __int64 *)*v19 != v19 + 2 )
          {
            v36 = v6;
            _libc_free(*v19);
            v6 = v36;
          }
        }
        while ( v19 != (unsigned __int64 *)v10 );
        v10 = *(_QWORD *)a1;
      }
      if ( v10 != a1 + 16 )
      {
        v37 = v6;
        _libc_free(v10);
        v6 = v37;
      }
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
