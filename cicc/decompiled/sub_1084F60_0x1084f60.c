// Function: sub_1084F60
// Address: 0x1084f60
//
__int64 __fastcall sub_1084F60(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rsi
  __int64 *v5; // rbx
  __int64 *v6; // r12
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rdi
  _QWORD *v10; // rbx
  _QWORD *v11; // r13
  _QWORD *v12; // r12
  _QWORD *v13; // rdi
  __int64 v14; // rdi
  _QWORD *v15; // rdi
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 *v18; // rbx
  __int64 *v19; // r12
  __int64 v20; // r13
  __int64 v21; // rdi
  __int64 v22; // rdi
  _QWORD *v23; // rbx
  _QWORD *v24; // r13
  _QWORD *v25; // r12
  _QWORD *v26; // rdi
  __int64 v27; // rdi
  _QWORD *v28; // rdi
  __int64 v29; // rdi

  v3 = a1[15];
  *a1 = &unk_49E6198;
  if ( v3 )
  {
    sub_C7D6A0(*(_QWORD *)(v3 + 216), 8LL * *(unsigned int *)(v3 + 232), 8);
    sub_C7D6A0(*(_QWORD *)(v3 + 184), 16LL * *(unsigned int *)(v3 + 200), 8);
    v4 = 16LL * *(unsigned int *)(v3 + 168);
    sub_C7D6A0(*(_QWORD *)(v3 + 152), v4, 8);
    sub_C0BF30(v3 + 96);
    v5 = *(__int64 **)(v3 + 80);
    v6 = *(__int64 **)(v3 + 72);
    if ( v5 != v6 )
    {
      do
      {
        v7 = *v6;
        if ( *v6 )
        {
          v8 = *(_QWORD *)(v7 + 64);
          if ( v8 != v7 + 80 )
            _libc_free(v8, v4);
          v9 = *(_QWORD *)(v7 + 24);
          if ( v9 != v7 + 48 )
            _libc_free(v9, v4);
          v4 = 136;
          j_j___libc_free_0(v7, 136);
        }
        ++v6;
      }
      while ( v5 != v6 );
      v6 = *(__int64 **)(v3 + 72);
    }
    if ( v6 )
    {
      v4 = *(_QWORD *)(v3 + 88) - (_QWORD)v6;
      j_j___libc_free_0(v6, v4);
    }
    v10 = *(_QWORD **)(v3 + 56);
    v11 = *(_QWORD **)(v3 + 48);
    if ( v10 != v11 )
    {
      do
      {
        v12 = (_QWORD *)*v11;
        if ( *v11 )
        {
          v13 = (_QWORD *)v12[15];
          if ( v13 != v12 + 17 )
            _libc_free(v13, v4);
          v14 = v12[12];
          if ( v14 )
            j_j___libc_free_0(v14, v12[14] - v14);
          v15 = (_QWORD *)v12[5];
          if ( v15 != v12 + 7 )
            j_j___libc_free_0(v15, v12[7] + 1LL);
          v4 = 144;
          j_j___libc_free_0(v12, 144);
        }
        ++v11;
      }
      while ( v10 != v11 );
      v11 = *(_QWORD **)(v3 + 48);
    }
    if ( v11 )
      j_j___libc_free_0(v11, *(_QWORD *)(v3 + 64) - (_QWORD)v11);
    a2 = 248;
    j_j___libc_free_0(v3, 248);
  }
  v16 = a1[14];
  if ( v16 )
  {
    sub_C7D6A0(*(_QWORD *)(v16 + 216), 8LL * *(unsigned int *)(v16 + 232), 8);
    sub_C7D6A0(*(_QWORD *)(v16 + 184), 16LL * *(unsigned int *)(v16 + 200), 8);
    v17 = 16LL * *(unsigned int *)(v16 + 168);
    sub_C7D6A0(*(_QWORD *)(v16 + 152), v17, 8);
    sub_C0BF30(v16 + 96);
    v18 = *(__int64 **)(v16 + 80);
    v19 = *(__int64 **)(v16 + 72);
    if ( v18 != v19 )
    {
      do
      {
        v20 = *v19;
        if ( *v19 )
        {
          v21 = *(_QWORD *)(v20 + 64);
          if ( v21 != v20 + 80 )
            _libc_free(v21, v17);
          v22 = *(_QWORD *)(v20 + 24);
          if ( v22 != v20 + 48 )
            _libc_free(v22, v17);
          v17 = 136;
          j_j___libc_free_0(v20, 136);
        }
        ++v19;
      }
      while ( v18 != v19 );
      v19 = *(__int64 **)(v16 + 72);
    }
    if ( v19 )
    {
      v17 = *(_QWORD *)(v16 + 88) - (_QWORD)v19;
      j_j___libc_free_0(v19, v17);
    }
    v23 = *(_QWORD **)(v16 + 56);
    v24 = *(_QWORD **)(v16 + 48);
    if ( v23 != v24 )
    {
      do
      {
        v25 = (_QWORD *)*v24;
        if ( *v24 )
        {
          v26 = (_QWORD *)v25[15];
          if ( v26 != v25 + 17 )
            _libc_free(v26, v17);
          v27 = v25[12];
          if ( v27 )
            j_j___libc_free_0(v27, v25[14] - v27);
          v28 = (_QWORD *)v25[5];
          if ( v28 != v25 + 7 )
            j_j___libc_free_0(v28, v25[7] + 1LL);
          v17 = 144;
          j_j___libc_free_0(v25, 144);
        }
        ++v24;
      }
      while ( v23 != v24 );
      v24 = *(_QWORD **)(v16 + 48);
    }
    if ( v24 )
      j_j___libc_free_0(v24, *(_QWORD *)(v16 + 64) - (_QWORD)v24);
    a2 = 248;
    j_j___libc_free_0(v16, 248);
  }
  v29 = a1[13];
  if ( v29 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v29 + 8LL))(v29);
  return sub_E8EC10((__int64)a1, a2);
}
