// Function: sub_3503E60
// Address: 0x3503e60
//
__int64 __fastcall sub_3503E60(__int64 *a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // r12
  __int64 *v15; // r15
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rcx
  _QWORD *v22; // r8
  __int64 v23; // r9
  unsigned __int64 v24; // rdi
  _QWORD *v25; // r8
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // r8
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // r15
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  unsigned __int64 v32; // [rsp+8h] [rbp-38h]
  unsigned __int64 v33; // [rsp+8h] [rbp-38h]

  v2 = sub_B82360(a1[1], (__int64)&unk_501EC08);
  if ( v2 )
  {
    v3 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_501EC08);
    if ( v3 )
      return v3 + 200;
  }
  v5 = (__int64 *)a1[1];
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_40:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_501F1C8 )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_40;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_501F1C8)
     + 169;
  v9 = sub_B82360(a1[1], (__int64)&unk_50208AC);
  if ( v9 && (v10 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v9 + 104LL))(v9, &unk_50208AC)) != 0 )
  {
    v11 = v10 + 200;
    v12 = sub_B82360(a1[1], (__int64)&unk_501FE44);
    if ( v12 )
      (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)v12 + 104LL))(v12, &unk_501FE44);
  }
  else
  {
    v16 = (__int64)&unk_501FE44;
    v17 = sub_B82360(a1[1], (__int64)&unk_501FE44);
    if ( !v17
      || (v16 = (__int64)&unk_501FE44,
          v18 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v17 + 104LL))(v17, &unk_501FE44),
          v19 = v18 + 200,
          !v18) )
    {
      v25 = (_QWORD *)sub_22077B0(0x80u);
      if ( v25 )
      {
        memset(v25, 0, 0x80u);
        *v25 = v25 + 2;
        v25[1] = 0x100000000LL;
        v25[3] = v25 + 5;
        v25[4] = 0x600000000LL;
      }
      v26 = a1[27];
      a1[27] = (__int64)v25;
      if ( v26 )
      {
        v27 = *(_QWORD *)(v26 + 24);
        v28 = v27 + 8LL * *(unsigned int *)(v26 + 32);
        if ( v27 != v28 )
        {
          do
          {
            v29 = *(_QWORD *)(v28 - 8);
            v28 -= 8LL;
            if ( v29 )
            {
              v30 = *(_QWORD *)(v29 + 24);
              if ( v30 != v29 + 40 )
              {
                v32 = v27;
                _libc_free(v30);
                v27 = v32;
              }
              v33 = v27;
              j_j___libc_free_0(v29);
              v27 = v33;
            }
          }
          while ( v27 != v28 );
          v27 = *(_QWORD *)(v26 + 24);
        }
        if ( v27 != v26 + 40 )
          _libc_free(v27);
        if ( *(_QWORD *)v26 != v26 + 16 )
          _libc_free(*(_QWORD *)v26);
        v16 = 128;
        j_j___libc_free_0(v26);
        v25 = (_QWORD *)a1[27];
      }
      v31 = a1[28];
      v25[13] = v31;
      *((_DWORD *)v25 + 30) = *(_DWORD *)(v31 + 120);
      sub_2E708A0((__int64)v25);
      v19 = a1[27];
    }
    v22 = (_QWORD *)sub_22077B0(0x98u);
    if ( v22 )
    {
      memset(v22, 0, 0x98u);
      v21 = 0;
      v22[18] = 1;
      v22[9] = v22 + 11;
      v22[10] = 0x400000000LL;
      v22[15] = v22 + 17;
    }
    v24 = a1[26];
    a1[26] = (__int64)v22;
    if ( v24 )
    {
      sub_3503B80(v24, v16);
      v22 = (_QWORD *)a1[26];
    }
    sub_2EA84A0((__int64)v22, v19, v20, v21, (__int64)v22, v23);
    v11 = a1[26];
  }
  v13 = (_QWORD *)sub_22077B0(8u);
  v14 = (__int64)v13;
  if ( v13 )
    sub_2E39A70(v13);
  v15 = (__int64 *)a1[25];
  a1[25] = v14;
  if ( v15 )
  {
    sub_2E39BC0(v15);
    j_j___libc_free_0((unsigned __int64)v15);
    v14 = a1[25];
  }
  sub_2E43BF0(v14, a1[28], v8, v11);
  return a1[25];
}
