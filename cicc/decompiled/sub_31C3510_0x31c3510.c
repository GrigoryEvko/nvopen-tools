// Function: sub_31C3510
// Address: 0x31c3510
//
void __fastcall sub_31C3510(__int64 a1, __int64 a2)
{
  char *v2; // r15
  __int64 v4; // r12
  unsigned __int64 v5; // rdx
  _QWORD *v6; // r14
  unsigned __int64 v7; // r9
  char *v8; // rbx
  _QWORD *v9; // rdx
  char *v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rdi
  _QWORD *v13; // rbx
  __int64 v14; // rdi
  _QWORD *v15; // rcx
  _QWORD *i; // rbx
  __int64 v17; // rdi
  char *v18; // r13
  __int64 v19; // rbx
  __int64 v20; // rdi
  _QWORD *v21; // rbx
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  _QWORD *v24; // rbx
  __int64 v25; // rdi
  int v26; // r15d
  _QWORD *v27; // rbx
  __int64 v28; // rdx
  _QWORD *v29; // rdi
  int v30; // [rsp-54h] [rbp-54h]
  __int64 v31; // [rsp-50h] [rbp-50h]
  unsigned __int64 v32; // [rsp-50h] [rbp-50h]
  unsigned __int64 v33; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != a2 )
  {
    v2 = (char *)(a2 + 16);
    v4 = a2;
    v5 = *(unsigned int *)(a1 + 8);
    v6 = *(_QWORD **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v7 = *(unsigned int *)(a2 + 8);
      v30 = *(_DWORD *)(a2 + 8);
      if ( v7 <= v5 )
      {
        v15 = *(_QWORD **)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v27 = &v6[v7];
          do
          {
            v28 = *(_QWORD *)v2;
            *(_QWORD *)v2 = 0;
            v29 = (_QWORD *)*v6;
            *v6 = v28;
            if ( v29 )
              (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD *))(*v29 + 8LL))(v29, a2, *v29, v15);
            ++v6;
            v2 += 8;
          }
          while ( v6 != v27 );
          v15 = *(_QWORD **)a1;
          v5 = *(unsigned int *)(a1 + 8);
        }
        for ( i = &v15[v5]; v6 != i; --i )
        {
          v17 = *(i - 1);
          if ( v17 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
        }
        *(_DWORD *)(a1 + 8) = v30;
        v18 = *(char **)a2;
        v19 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v19 )
        {
          do
          {
            v20 = *(_QWORD *)(v19 - 8);
            v19 -= 8;
            if ( v20 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
          }
          while ( v18 != (char *)v19 );
        }
      }
      else
      {
        if ( v7 > *(unsigned int *)(a1 + 12) )
        {
          v24 = &v6[v5];
          while ( v6 != v24 )
          {
            while ( 1 )
            {
              v25 = *--v24;
              if ( !v25 )
                break;
              v32 = v7;
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
              v7 = v32;
              if ( v6 == v24 )
                goto LABEL_43;
            }
          }
LABEL_43:
          *(_DWORD *)(a1 + 8) = 0;
          a2 = sub_C8D7D0(a1, a1 + 16, v7, 8u, &v33, v7);
          v6 = (_QWORD *)a2;
          sub_31C3490(a1, (_QWORD *)a2);
          v26 = v33;
          if ( a1 + 16 != *(_QWORD *)a1 )
            _libc_free(*(_QWORD *)a1);
          *(_QWORD *)a1 = a2;
          *(_DWORD *)(a1 + 12) = v26;
          v2 = *(char **)v4;
          v7 = *(unsigned int *)(v4 + 8);
          v8 = *(char **)v4;
        }
        else
        {
          v8 = (char *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v31 = 8 * v5;
            v21 = &v6[v5];
            do
            {
              v22 = *(_QWORD *)v2;
              *(_QWORD *)v2 = 0;
              v23 = (_QWORD *)*v6;
              *v6 = v22;
              if ( v23 )
                (*(void (__fastcall **)(_QWORD *))(*v23 + 8LL))(v23);
              ++v6;
              v2 += 8;
            }
            while ( v6 != v21 );
            v2 = *(char **)a2;
            v7 = *(unsigned int *)(a2 + 8);
            v6 = (_QWORD *)(v31 + *(_QWORD *)a1);
            v8 = (char *)(*(_QWORD *)a2 + v31);
          }
        }
        v9 = (_QWORD *)((char *)v6 + &v2[8 * v7] - v8);
        if ( &v2[8 * v7] != v8 )
        {
          do
          {
            if ( v6 )
            {
              *v6 = *(_QWORD *)v8;
              *(_QWORD *)v8 = 0;
            }
            ++v6;
            v8 += 8;
          }
          while ( v6 != v9 );
        }
        *(_DWORD *)(a1 + 8) = v30;
        v10 = *(char **)v4;
        v11 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
        if ( *(_QWORD *)v4 != v11 )
        {
          do
          {
            v12 = *(_QWORD *)(v11 - 8);
            v11 -= 8;
            if ( v12 )
              (*(void (__fastcall **)(__int64, __int64, _QWORD *))(*(_QWORD *)v12 + 8LL))(v12, a2, v9);
          }
          while ( v10 != (char *)v11 );
        }
      }
      *(_DWORD *)(v4 + 8) = 0;
    }
    else
    {
      v13 = &v6[v5];
      if ( v13 != v6 )
      {
        do
        {
          v14 = *--v13;
          if ( v14 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
        }
        while ( v13 != v6 );
        v6 = *(_QWORD **)a1;
      }
      if ( v6 != (_QWORD *)(a1 + 16) )
        _libc_free((unsigned __int64)v6);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v2;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
