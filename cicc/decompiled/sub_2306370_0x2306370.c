// Function: sub_2306370
// Address: 0x2306370
//
__int64 *__fastcall sub_2306370(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r14
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // r13
  unsigned __int64 v14; // rdi
  _QWORD *v16; // rdx
  _QWORD *v17; // rbx
  unsigned __int64 v18; // rdi
  __int64 v19; // [rsp+0h] [rbp-90h]
  __int64 v20; // [rsp+8h] [rbp-88h]
  unsigned __int64 v21; // [rsp+10h] [rbp-80h]
  unsigned int v22; // [rsp+18h] [rbp-78h]
  _QWORD *v23; // [rsp+18h] [rbp-78h]
  __int64 v24; // [rsp+20h] [rbp-70h] BYREF
  __int64 v25; // [rsp+28h] [rbp-68h]
  _QWORD *v26; // [rsp+30h] [rbp-60h]
  __int64 v27; // [rsp+38h] [rbp-58h]
  unsigned int v28; // [rsp+40h] [rbp-50h]
  unsigned __int64 v29; // [rsp+48h] [rbp-48h]
  __int64 v30; // [rsp+50h] [rbp-40h]
  __int64 v31; // [rsp+58h] [rbp-38h]

  sub_2C725F0(&v24, a2 + 8);
  v3 = v27;
  v27 = 0;
  v4 = (__int64)v26;
  v5 = v30;
  v21 = v29;
  ++v25;
  v6 = v31;
  v19 = v24;
  v22 = v28;
  v20 = v3;
  v26 = 0;
  v28 = 0;
  v31 = 0;
  v30 = 0;
  v29 = 0;
  v7 = sub_22077B0(0x48u);
  v8 = v7;
  if ( v7 )
  {
    *(_QWORD *)(v7 + 48) = v21;
    *(_QWORD *)(v7 + 16) = 1;
    *(_QWORD *)(v7 + 8) = v19;
    *(_QWORD *)v7 = &unk_4A0ACC8;
    *(_QWORD *)(v7 + 32) = v20;
    *(_DWORD *)(v7 + 40) = v22;
    *(_QWORD *)(v7 + 64) = v6;
    *(_QWORD *)(v7 + 24) = v4;
    v4 = 0;
    *(_QWORD *)(v7 + 56) = v5;
    v9 = 0;
  }
  else
  {
    v9 = 88LL * v22;
    if ( v21 )
      j_j___libc_free_0(v21);
    v16 = (_QWORD *)(v4 + 88LL * v22);
    v17 = (_QWORD *)v4;
    if ( v22 )
    {
      do
      {
        if ( *v17 != -8192 && *v17 != -4096 )
        {
          v18 = v17[2];
          if ( (_QWORD *)v18 != v17 + 4 )
          {
            v23 = v16;
            _libc_free(v18);
            v16 = v23;
          }
        }
        v17 += 11;
      }
      while ( v16 != v17 );
    }
  }
  sub_C7D6A0(v4, v9, 8);
  v10 = v29;
  *a1 = v8;
  if ( v10 )
    j_j___libc_free_0(v10);
  v11 = v28;
  if ( v28 )
  {
    v12 = v26;
    v13 = &v26[11 * v28];
    do
    {
      if ( *v12 != -8192 && *v12 != -4096 )
      {
        v14 = v12[2];
        if ( (_QWORD *)v14 != v12 + 4 )
          _libc_free(v14);
      }
      v12 += 11;
    }
    while ( v13 != v12 );
    v11 = v28;
  }
  sub_C7D6A0((__int64)v26, 88 * v11, 8);
  return a1;
}
