// Function: sub_E98B30
// Address: 0xe98b30
//
__int64 __fastcall sub_E98B30(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 *v4; // r14
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdi
  __int64 v15; // rdi
  _QWORD *v16; // rdi
  _QWORD *v17; // r14
  _QWORD *v18; // r13
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  _QWORD *v21; // rdi
  __int64 v22; // rdi
  __int64 result; // rax
  __int64 v24; // r12
  __int64 (__fastcall *v25)(__int64); // rax
  __int64 *v27; // [rsp+18h] [rbp-38h]
  __int64 v28; // [rsp+18h] [rbp-38h]

  *a1 = &unk_49E3C78;
  v3 = (_QWORD *)a1[15];
  if ( v3 != a1 + 17 )
    _libc_free(v3, a2);
  v4 = (__int64 *)a1[10];
  v27 = (__int64 *)a1[11];
  if ( v27 != v4 )
  {
    do
    {
      v5 = *v4;
      if ( *v4 )
      {
        v6 = *(_QWORD *)(v5 + 168);
        v7 = *(_QWORD *)(v5 + 160);
        if ( v6 != v7 )
        {
          do
          {
            v8 = *(_QWORD *)(v7 + 64);
            v9 = v7 + 80;
            if ( v8 != v7 + 80 )
              _libc_free(v8, a2);
            v10 = *(unsigned int *)(v7 + 56);
            v11 = *(_QWORD *)(v7 + 40);
            v7 += 80;
            a2 = 16 * v10;
            sub_C7D6A0(v11, a2, 8);
          }
          while ( v6 != v9 );
          v7 = *(_QWORD *)(v5 + 160);
        }
        if ( v7 )
        {
          a2 = *(_QWORD *)(v5 + 176) - v7;
          j_j___libc_free_0(v7, a2);
        }
        v12 = *(_QWORD *)(v5 + 144);
        v13 = v12 + 48LL * *(unsigned int *)(v5 + 152);
        if ( v12 != v13 )
        {
          do
          {
            v14 = *(_QWORD *)(v13 - 40);
            v13 -= 48;
            if ( v14 )
            {
              a2 = *(_QWORD *)(v13 + 24) - v14;
              j_j___libc_free_0(v14, a2);
            }
          }
          while ( v12 != v13 );
          v13 = *(_QWORD *)(v5 + 144);
        }
        if ( v5 + 160 != v13 )
          _libc_free(v13, a2);
        sub_C7D6A0(*(_QWORD *)(v5 + 120), 16LL * *(unsigned int *)(v5 + 136), 8);
        v15 = *(_QWORD *)(v5 + 88);
        if ( v15 )
          j_j___libc_free_0(v15, *(_QWORD *)(v5 + 104) - v15);
        a2 = 184;
        j_j___libc_free_0(v5, 184);
      }
      ++v4;
    }
    while ( v27 != v4 );
    v4 = (__int64 *)a1[10];
  }
  if ( v4 )
  {
    v28 = a1[12];
    a2 = v28 - (_QWORD)v4;
    j_j___libc_free_0(v4, v28 - (_QWORD)v4);
  }
  v16 = (_QWORD *)a1[6];
  if ( v16 != a1 + 8 )
    _libc_free(v16, a2);
  v17 = (_QWORD *)a1[4];
  v18 = (_QWORD *)a1[3];
  if ( v17 != v18 )
  {
    do
    {
      v19 = (_QWORD *)v18[5];
      v20 = (_QWORD *)v18[4];
      if ( v19 != v20 )
      {
        do
        {
          v21 = (_QWORD *)v20[9];
          if ( v21 != v20 + 11 )
            j_j___libc_free_0(v21, v20[11] + 1LL);
          v22 = v20[6];
          if ( v22 )
            j_j___libc_free_0(v22, v20[8] - v22);
          v20 += 13;
        }
        while ( v19 != v20 );
        v20 = (_QWORD *)v18[4];
      }
      if ( v20 )
        j_j___libc_free_0(v20, v18[6] - (_QWORD)v20);
      v18 += 12;
    }
    while ( v17 != v18 );
    v18 = (_QWORD *)a1[3];
  }
  if ( v18 )
    j_j___libc_free_0(v18, a1[5] - (_QWORD)v18);
  result = (__int64)a1;
  v24 = a1[2];
  if ( v24 )
  {
    v25 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL);
    if ( v25 == sub_E977D0 )
    {
      nullsub_339();
      return j_j___libc_free_0(v24, 16);
    }
    else
    {
      return v25(a1[2]);
    }
  }
  return result;
}
