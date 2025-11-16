// Function: sub_31F9F70
// Address: 0x31f9f70
//
void __fastcall sub_31F9F70(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  _QWORD *v7; // r15
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // rbx
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v18; // rbx
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // r14
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  _QWORD *v27; // r15
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // rdi
  __int64 v30; // rbx
  unsigned __int64 v31; // r14
  unsigned __int64 v32; // r9
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdi
  void *v35; // rdi
  bool v36; // cc
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  __int64 v41; // [rsp+8h] [rbp-48h]
  unsigned __int64 v42; // [rsp+10h] [rbp-40h]
  unsigned __int64 v43; // [rsp+10h] [rbp-40h]
  unsigned __int64 v44; // [rsp+18h] [rbp-38h]
  unsigned __int64 v45; // [rsp+18h] [rbp-38h]

  v41 = a2;
  while ( a1 != v41 )
  {
    v41 -= 16;
    v2 = *(_QWORD *)(v41 + 8);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 416);
      if ( v3 )
        j_j___libc_free_0(v3);
      v4 = *(_QWORD *)(v2 + 392);
      if ( v4 )
        j_j___libc_free_0(v4);
      v5 = *(_QWORD *)(v2 + 368);
      if ( v5 )
        j_j___libc_free_0(v5);
      v6 = *(_QWORD *)(v2 + 344);
      if ( v6 != v2 + 360 )
        _libc_free(v6);
      v7 = *(_QWORD **)(v2 + 304);
      while ( v7 )
      {
        v8 = (unsigned __int64)v7;
        v7 = (_QWORD *)*v7;
        v9 = *(_QWORD *)(v8 + 152);
        if ( v9 != v8 + 168 )
          _libc_free(v9);
        v10 = *(_QWORD *)(v8 + 120);
        if ( v10 != v8 + 136 )
          _libc_free(v10);
        v11 = *(_QWORD *)(v8 + 16);
        v12 = v11 + 88LL * *(unsigned int *)(v8 + 24);
        if ( v11 != v12 )
        {
          do
          {
            v12 -= 88LL;
            if ( *(_BYTE *)(v12 + 80) )
            {
              v36 = *(_DWORD *)(v12 + 72) <= 0x40u;
              *(_BYTE *)(v12 + 80) = 0;
              if ( !v36 )
              {
                v37 = *(_QWORD *)(v12 + 64);
                if ( v37 )
                  j_j___libc_free_0_0(v37);
              }
            }
            v13 = *(_QWORD *)(v12 + 40);
            v14 = v13 + 40LL * *(unsigned int *)(v12 + 48);
            if ( v13 != v14 )
            {
              do
              {
                v14 -= 40LL;
                v15 = *(_QWORD *)(v14 + 8);
                if ( v15 != v14 + 24 )
                {
                  v42 = v13;
                  v44 = v14;
                  _libc_free(v15);
                  v13 = v42;
                  v14 = v44;
                }
              }
              while ( v13 != v14 );
              v13 = *(_QWORD *)(v12 + 40);
            }
            if ( v13 != v12 + 56 )
              _libc_free(v13);
            sub_C7D6A0(*(_QWORD *)(v12 + 16), 12LL * *(unsigned int *)(v12 + 32), 4);
          }
          while ( v11 != v12 );
          v12 = *(_QWORD *)(v8 + 16);
        }
        if ( v12 != v8 + 32 )
          _libc_free(v12);
        j_j___libc_free_0(v8);
      }
      memset(*(void **)(v2 + 288), 0, 8LL * *(_QWORD *)(v2 + 296));
      v16 = *(_QWORD *)(v2 + 288);
      *(_QWORD *)(v2 + 312) = 0;
      *(_QWORD *)(v2 + 304) = 0;
      if ( v16 != v2 + 336 )
        j_j___libc_free_0(v16);
      v17 = *(_QWORD *)(v2 + 256);
      if ( v17 != v2 + 272 )
        _libc_free(v17);
      v18 = *(_QWORD *)(v2 + 152);
      v19 = v18 + 88LL * *(unsigned int *)(v2 + 160);
      if ( v18 != v19 )
      {
        do
        {
          v19 -= 88LL;
          if ( *(_BYTE *)(v19 + 80) )
          {
            v36 = *(_DWORD *)(v19 + 72) <= 0x40u;
            *(_BYTE *)(v19 + 80) = 0;
            if ( !v36 )
            {
              v39 = *(_QWORD *)(v19 + 64);
              if ( v39 )
                j_j___libc_free_0_0(v39);
            }
          }
          v20 = *(_QWORD *)(v19 + 40);
          v21 = v20 + 40LL * *(unsigned int *)(v19 + 48);
          if ( v20 != v21 )
          {
            do
            {
              v21 -= 40LL;
              v22 = *(_QWORD *)(v21 + 8);
              if ( v22 != v21 + 24 )
                _libc_free(v22);
            }
            while ( v20 != v21 );
            v20 = *(_QWORD *)(v19 + 40);
          }
          if ( v20 != v19 + 56 )
            _libc_free(v20);
          sub_C7D6A0(*(_QWORD *)(v19 + 16), 12LL * *(unsigned int *)(v19 + 32), 4);
        }
        while ( v18 != v19 );
        v19 = *(_QWORD *)(v2 + 152);
      }
      if ( v19 != v2 + 168 )
        _libc_free(v19);
      v23 = *(_QWORD *)(v2 + 120);
      while ( v23 )
      {
        sub_31F5020(*(_QWORD *)(v23 + 24));
        v24 = v23;
        v23 = *(_QWORD *)(v23 + 16);
        j_j___libc_free_0(v24);
      }
      v25 = *(_QWORD *)(v2 + 80);
      if ( v25 != v2 + 96 )
        _libc_free(v25);
      v26 = *(_QWORD *)(v2 + 56);
      if ( v26 != v2 + 72 )
        _libc_free(v26);
      v27 = *(_QWORD **)(v2 + 16);
      while ( v27 )
      {
        v28 = (unsigned __int64)v27;
        v27 = (_QWORD *)*v27;
        v29 = *(_QWORD *)(v28 + 120);
        if ( v29 != v28 + 136 )
          _libc_free(v29);
        v30 = *(_QWORD *)(v28 + 16);
        v31 = v30 + 88LL * *(unsigned int *)(v28 + 24);
        if ( v30 != v31 )
        {
          do
          {
            v31 -= 88LL;
            if ( *(_BYTE *)(v31 + 80) )
            {
              v36 = *(_DWORD *)(v31 + 72) <= 0x40u;
              *(_BYTE *)(v31 + 80) = 0;
              if ( !v36 )
              {
                v38 = *(_QWORD *)(v31 + 64);
                if ( v38 )
                  j_j___libc_free_0_0(v38);
              }
            }
            v32 = *(_QWORD *)(v31 + 40);
            v33 = v32 + 40LL * *(unsigned int *)(v31 + 48);
            if ( v32 != v33 )
            {
              do
              {
                v33 -= 40LL;
                v34 = *(_QWORD *)(v33 + 8);
                if ( v34 != v33 + 24 )
                {
                  v43 = v32;
                  v45 = v33;
                  _libc_free(v34);
                  v32 = v43;
                  v33 = v45;
                }
              }
              while ( v32 != v33 );
              v32 = *(_QWORD *)(v31 + 40);
            }
            if ( v32 != v31 + 56 )
              _libc_free(v32);
            sub_C7D6A0(*(_QWORD *)(v31 + 16), 12LL * *(unsigned int *)(v31 + 32), 4);
          }
          while ( v30 != v31 );
          v31 = *(_QWORD *)(v28 + 16);
        }
        if ( v31 != v28 + 32 )
          _libc_free(v31);
        j_j___libc_free_0(v28);
      }
      memset(*(void **)v2, 0, 8LL * *(_QWORD *)(v2 + 8));
      v35 = *(void **)v2;
      *(_QWORD *)(v2 + 24) = 0;
      *(_QWORD *)(v2 + 16) = 0;
      if ( v35 != (void *)(v2 + 48) )
        j_j___libc_free_0((unsigned __int64)v35);
      j_j___libc_free_0(v2);
    }
  }
}
