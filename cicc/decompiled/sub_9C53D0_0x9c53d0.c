// Function: sub_9C53D0
// Address: 0x9c53d0
//
void __fastcall sub_9C53D0(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // r15
  _QWORD *v4; // r12
  __int64 *v5; // rbx
  __int64 *v6; // r14
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rdi
  __int64 *v11; // r14
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 *v16; // r15
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  _QWORD *v24; // r14
  __int64 v25; // r15
  __int64 v26; // r12
  __int64 v27; // rdi
  __int64 v28; // r15
  __int64 v29; // r12
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // rdi
  _QWORD *v33; // rdi
  _QWORD *v34; // rdi
  _QWORD *v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  v3 = (_QWORD *)a1[13];
  *a1 = &unk_49D97B0;
  if ( v3 )
  {
    v4 = (_QWORD *)*v3;
    v35 = (_QWORD *)v3[1];
    if ( v35 != (_QWORD *)*v3 )
    {
      do
      {
        v5 = (__int64 *)v4[12];
        v6 = (__int64 *)v4[11];
        if ( v5 != v6 )
        {
          do
          {
            v7 = *v6;
            if ( *v6 )
            {
              a2 = v6[2] - v7;
              j_j___libc_free_0(v7, a2);
            }
            v6 += 3;
          }
          while ( v5 != v6 );
          v6 = (__int64 *)v4[11];
        }
        if ( v6 )
        {
          a2 = v4[13] - (_QWORD)v6;
          j_j___libc_free_0(v6, a2);
        }
        v8 = v4[9];
        v9 = v4[8];
        if ( v8 != v9 )
        {
          do
          {
            v10 = *(_QWORD *)(v9 + 8);
            if ( v10 != v9 + 24 )
              _libc_free(v10, a2);
            v9 += 72;
          }
          while ( v8 != v9 );
          v9 = v4[8];
        }
        if ( v9 )
        {
          a2 = v4[10] - v9;
          j_j___libc_free_0(v9, a2);
        }
        if ( (_QWORD *)*v4 != v4 + 3 )
          _libc_free(*v4, a2);
        v4 += 14;
      }
      while ( v35 != v4 );
      v4 = (_QWORD *)*v3;
    }
    if ( v4 )
      j_j___libc_free_0(v4, v3[2] - (_QWORD)v4);
    a2 = 24;
    j_j___libc_free_0(v3, 24);
  }
  v11 = (__int64 *)a1[12];
  if ( v11 )
  {
    v12 = v11[1];
    v13 = *v11;
    if ( v12 != *v11 )
    {
      do
      {
        v14 = *(_QWORD *)(v13 + 72);
        if ( v14 != v13 + 88 )
          _libc_free(v14, a2);
        v15 = *(_QWORD *)(v13 + 8);
        if ( v15 != v13 + 24 )
          _libc_free(v15, a2);
        v13 += 136;
      }
      while ( v12 != v13 );
      v13 = *v11;
    }
    if ( v13 )
      j_j___libc_free_0(v13, v11[2] - v13);
    a2 = 24;
    j_j___libc_free_0(v11, 24);
  }
  v16 = (__int64 *)a1[11];
  if ( v16 )
  {
    v17 = *v16;
    v36 = v16[1];
    if ( v36 != *v16 )
    {
      do
      {
        v18 = *(_QWORD *)(v17 + 48);
        v19 = *(_QWORD *)(v17 + 40);
        if ( v18 != v19 )
        {
          do
          {
            if ( *(_DWORD *)(v19 + 40) > 0x40u )
            {
              v20 = *(_QWORD *)(v19 + 32);
              if ( v20 )
                j_j___libc_free_0_0(v20);
            }
            if ( *(_DWORD *)(v19 + 24) > 0x40u )
            {
              v21 = *(_QWORD *)(v19 + 16);
              if ( v21 )
                j_j___libc_free_0_0(v21);
            }
            v19 += 48;
          }
          while ( v18 != v19 );
          v19 = *(_QWORD *)(v17 + 40);
        }
        if ( v19 )
          j_j___libc_free_0(v19, *(_QWORD *)(v17 + 56) - v19);
        if ( *(_DWORD *)(v17 + 32) > 0x40u )
        {
          v22 = *(_QWORD *)(v17 + 24);
          if ( v22 )
            j_j___libc_free_0_0(v22);
        }
        if ( *(_DWORD *)(v17 + 16) > 0x40u )
        {
          v23 = *(_QWORD *)(v17 + 8);
          if ( v23 )
            j_j___libc_free_0_0(v23);
        }
        v17 += 64;
      }
      while ( v36 != v17 );
      v17 = *v16;
    }
    if ( v17 )
      j_j___libc_free_0(v17, v16[2] - v17);
    a2 = 24;
    j_j___libc_free_0(v16, 24);
  }
  v24 = (_QWORD *)a1[10];
  if ( v24 )
  {
    v25 = v24[13];
    v26 = v24[12];
    if ( v25 != v26 )
    {
      do
      {
        v27 = *(_QWORD *)(v26 + 16);
        if ( v27 )
          j_j___libc_free_0(v27, *(_QWORD *)(v26 + 32) - v27);
        v26 += 40;
      }
      while ( v25 != v26 );
      v26 = v24[12];
    }
    if ( v26 )
      j_j___libc_free_0(v26, v24[14] - v26);
    v28 = v24[10];
    v29 = v24[9];
    if ( v28 != v29 )
    {
      do
      {
        v30 = *(_QWORD *)(v29 + 16);
        if ( v30 )
          j_j___libc_free_0(v30, *(_QWORD *)(v29 + 32) - v30);
        v29 += 40;
      }
      while ( v28 != v29 );
      v29 = v24[9];
    }
    if ( v29 )
      j_j___libc_free_0(v29, v24[11] - v29);
    v31 = v24[6];
    if ( v31 )
      j_j___libc_free_0(v31, v24[8] - v31);
    v32 = v24[3];
    if ( v32 )
      j_j___libc_free_0(v32, v24[5] - v32);
    if ( *v24 )
      j_j___libc_free_0(*v24, v24[2] - *v24);
    a2 = 120;
    j_j___libc_free_0(v24, 120);
  }
  v33 = (_QWORD *)a1[8];
  if ( a1 + 10 != v33 )
    _libc_free(v33, a2);
  v34 = (_QWORD *)a1[5];
  if ( v34 != a1 + 7 )
    _libc_free(v34, a2);
}
