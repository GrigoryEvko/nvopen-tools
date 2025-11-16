// Function: sub_3938C60
// Address: 0x3938c60
//
void __fastcall sub_3938C60(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v2; // rdi
  _QWORD *v3; // rdx
  void (__fastcall *v4)(_QWORD *); // rax
  unsigned __int64 v5; // rax
  unsigned __int64 *v6; // r14
  unsigned __int64 v7; // r13
  _QWORD *v8; // rbx
  _QWORD *v9; // r12
  _QWORD *v10; // r15
  unsigned __int64 v11; // rdi
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  _QWORD *v14; // r15
  unsigned __int64 v15; // rdi
  __int64 v16; // rdi
  unsigned __int64 v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rbx
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // [rsp+0h] [rbp-50h]
  unsigned __int64 v27; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v28; // [rsp+10h] [rbp-40h]

  v1 = a1[5];
  *a1 = &unk_4A3EF90;
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 8);
    if ( v2 )
      j_j___libc_free_0(v2);
    j_j___libc_free_0(v1);
  }
  v3 = (_QWORD *)a1[4];
  v27 = (unsigned __int64)v3;
  if ( v3 )
  {
    v4 = *(void (__fastcall **)(_QWORD *))(*v3 + 8LL);
    if ( v4 == sub_3938740 )
    {
      *v3 = &unk_4A3EF30;
      v5 = v3[1];
      v26 = v5;
      if ( v5 )
      {
        v6 = *(unsigned __int64 **)(v5 + 32);
        v28 = *(unsigned __int64 **)(v5 + 40);
        if ( v28 != v6 )
        {
          do
          {
            v7 = v6[3];
            if ( v7 )
            {
              v8 = *(_QWORD **)(v7 + 32);
              v9 = *(_QWORD **)(v7 + 24);
              if ( v8 != v9 )
              {
                do
                {
                  v10 = (_QWORD *)*v9;
                  while ( v9 != v10 )
                  {
                    v11 = (unsigned __int64)v10;
                    v10 = (_QWORD *)*v10;
                    j_j___libc_free_0(v11);
                  }
                  v9 += 3;
                }
                while ( v8 != v9 );
                v9 = *(_QWORD **)(v7 + 24);
              }
              if ( v9 )
                j_j___libc_free_0((unsigned __int64)v9);
              v12 = *(_QWORD **)(v7 + 8);
              v13 = *(_QWORD **)v7;
              if ( v12 != *(_QWORD **)v7 )
              {
                do
                {
                  v14 = (_QWORD *)*v13;
                  while ( v13 != v14 )
                  {
                    v15 = (unsigned __int64)v14;
                    v14 = (_QWORD *)*v14;
                    j_j___libc_free_0(v15);
                  }
                  v13 += 3;
                }
                while ( v12 != v13 );
                v13 = *(_QWORD **)v7;
              }
              if ( v13 )
                j_j___libc_free_0((unsigned __int64)v13);
              j_j___libc_free_0(v7);
            }
            if ( *v6 )
              j_j___libc_free_0(*v6);
            v6 += 7;
          }
          while ( v28 != v6 );
          v6 = *(unsigned __int64 **)(v26 + 32);
        }
        if ( v6 )
          j_j___libc_free_0((unsigned __int64)v6);
        j_j___libc_free_0(v26);
      }
      j_j___libc_free_0(v27);
    }
    else
    {
      v4(v3);
    }
  }
  v16 = a1[3];
  if ( v16 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
  v17 = a1[2];
  *a1 = &unk_4A3EEF0;
  if ( v17 )
  {
    v18 = *(_QWORD *)(v17 + 104);
    if ( v18 )
      j_j___libc_free_0(v18);
    v19 = *(_QWORD *)(v17 + 80);
    if ( v19 )
      j_j___libc_free_0(v19);
    v20 = *(_QWORD *)(v17 + 56);
    if ( v20 )
      j_j___libc_free_0(v20);
    v21 = *(_QWORD *)(v17 + 24);
    if ( *(_DWORD *)(v17 + 36) )
    {
      v22 = *(unsigned int *)(v17 + 32);
      if ( (_DWORD)v22 )
      {
        v23 = 8 * v22;
        v24 = 0;
        do
        {
          v25 = *(_QWORD *)(v21 + v24);
          if ( v25 != -8 && v25 )
          {
            _libc_free(v25);
            v21 = *(_QWORD *)(v17 + 24);
          }
          v24 += 8;
        }
        while ( v23 != v24 );
      }
    }
    _libc_free(v21);
    j_j___libc_free_0(v17);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
