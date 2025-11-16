// Function: sub_39A0390
// Address: 0x39a0390
//
void __fastcall sub_39A0390(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r13
  __int64 v5; // r15
  _QWORD *v6; // r12
  _QWORD *v7; // r15
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // r15
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r8
  __int64 v15; // r15
  __int64 v16; // r15
  __int64 v17; // r12
  unsigned __int64 v18; // rdi
  __int64 v19; // r15
  unsigned __int64 v20; // r8
  __int64 v21; // r15
  __int64 v22; // r12
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  _QWORD *v25; // r12
  _QWORD *v26; // r15
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // [rsp-40h] [rbp-40h]

  if ( a2 != a1 )
  {
    v3 = a2;
    do
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8;
      if ( v4 )
      {
        v5 = *(unsigned int *)(v4 + 920);
        *(_QWORD *)v4 = &unk_4A40688;
        if ( (_DWORD)v5 )
        {
          v6 = *(_QWORD **)(v4 + 904);
          v7 = &v6[2 * v5];
          do
          {
            if ( *v6 != -16 && *v6 != -8 )
            {
              v8 = v6[1];
              if ( v8 )
              {
                v9 = *(_QWORD *)(v8 + 40);
                if ( v9 != v8 + 56 )
                {
                  v28 = v6[1];
                  _libc_free(v9);
                  v8 = v28;
                }
                j_j___libc_free_0(v8);
              }
            }
            v6 += 2;
          }
          while ( v7 != v6 );
        }
        j___libc_free_0(*(_QWORD *)(v4 + 904));
        j___libc_free_0(*(_QWORD *)(v4 + 872));
        v10 = *(_QWORD *)(v4 + 808);
        if ( v10 != v4 + 824 )
          _libc_free(v10);
        v11 = *(_QWORD *)(v4 + 736);
        v12 = v11 + 56LL * *(unsigned int *)(v4 + 744);
        if ( v11 != v12 )
        {
          do
          {
            v12 -= 56LL;
            v13 = *(_QWORD *)(v12 + 8);
            if ( v13 != v12 + 24 )
              _libc_free(v13);
          }
          while ( v11 != v12 );
          v12 = *(_QWORD *)(v4 + 736);
        }
        if ( v12 != v4 + 752 )
          _libc_free(v12);
        v14 = *(_QWORD *)(v4 + 704);
        if ( *(_DWORD *)(v4 + 716) )
        {
          v15 = *(unsigned int *)(v4 + 712);
          if ( (_DWORD)v15 )
          {
            v16 = 8 * v15;
            v17 = 0;
            do
            {
              v18 = *(_QWORD *)(v14 + v17);
              if ( v18 && v18 != -8 )
              {
                _libc_free(v18);
                v14 = *(_QWORD *)(v4 + 704);
              }
              v17 += 8;
            }
            while ( v16 != v17 );
          }
        }
        _libc_free(v14);
        if ( *(_DWORD *)(v4 + 684) )
        {
          v19 = *(unsigned int *)(v4 + 680);
          v20 = *(_QWORD *)(v4 + 672);
          if ( (_DWORD)v19 )
          {
            v21 = 8 * v19;
            v22 = 0;
            do
            {
              v23 = *(_QWORD *)(v20 + v22);
              if ( v23 && v23 != -8 )
              {
                _libc_free(v23);
                v20 = *(_QWORD *)(v4 + 672);
              }
              v22 += 8;
            }
            while ( v21 != v22 );
          }
        }
        else
        {
          v20 = *(_QWORD *)(v4 + 672);
        }
        _libc_free(v20);
        v24 = *(unsigned int *)(v4 + 664);
        if ( (_DWORD)v24 )
        {
          v25 = *(_QWORD **)(v4 + 648);
          v26 = &v25[11 * v24];
          do
          {
            if ( *v25 != -8 && *v25 != -16 )
            {
              v27 = v25[1];
              if ( (_QWORD *)v27 != v25 + 3 )
                _libc_free(v27);
            }
            v25 += 11;
          }
          while ( v26 != v25 );
        }
        j___libc_free_0(*(_QWORD *)(v4 + 648));
        sub_39A20E0(v4);
        j_j___libc_free_0(v4);
      }
    }
    while ( a1 != v3 );
  }
}
