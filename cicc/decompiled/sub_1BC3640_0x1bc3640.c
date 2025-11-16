// Function: sub_1BC3640
// Address: 0x1bc3640
//
__int64 *__fastcall sub_1BC3640(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // rcx
  __int64 *v4; // r12
  __int64 v5; // r13
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rsi
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rbx
  _QWORD *v12; // rax
  __int64 *v13; // rax
  __int64 i; // rbx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *j; // r12
  __int64 *v18; // r14
  __int64 *v19; // r15
  __int64 v20; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+10h] [rbp-60h]
  __int64 *v26; // [rsp+20h] [rbp-50h]
  __int64 *v27; // [rsp+20h] [rbp-50h]
  __int64 *v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  __int64 *v30; // [rsp+28h] [rbp-48h]
  __int64 v31; // [rsp+30h] [rbp-40h]
  __int64 *v32; // [rsp+38h] [rbp-38h]

  v3 = a2;
  v4 = a2;
  v5 = a1[1];
  v32 = (__int64 *)*a1;
  v6 = 0x84BDA12F684BDA13LL * ((v5 - *a1) >> 3);
  if ( v6 == 0x97B425ED097B42LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0x84BDA12F684BDA13LL * ((v5 - *a1) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x7B425ED097B425EDLL * ((v5 - *a1) >> 3);
  v10 = (char *)v3 - (char *)v32;
  if ( v8 )
  {
    v22 = 0x7FFFFFFFFFFFFFB0LL;
LABEL_37:
    v24 = a3;
    v27 = v3;
    v29 = (char *)v3 - (char *)v32;
    v23 = sub_22077B0(v22);
    v10 = v29;
    v3 = v27;
    v31 = v23;
    a3 = v24;
    v25 = v23 + v22;
    v11 = v23 + 216;
    goto LABEL_7;
  }
  if ( v9 )
  {
    if ( v9 > 0x97B425ED097B42LL )
      v9 = 0x97B425ED097B42LL;
    v22 = 216 * v9;
    goto LABEL_37;
  }
  v25 = 0;
  v11 = 216;
  v31 = 0;
LABEL_7:
  v12 = (_QWORD *)(v31 + v10);
  if ( v31 + v10 )
  {
    *v12 = *(_QWORD *)a3;
    v12[1] = v12 + 3;
    v12[2] = 0x800000000LL;
    if ( *(_DWORD *)(a3 + 16) )
    {
      v30 = v3;
      sub_1BC1780((__int64)(v12 + 1), a3 + 8);
      v3 = v30;
    }
  }
  v13 = v32;
  if ( v3 != v32 )
  {
    for ( i = v31; ; i += 216 )
    {
      if ( i )
      {
        v15 = *v13;
        *(_DWORD *)(i + 16) = 0;
        *(_DWORD *)(i + 20) = 8;
        *(_QWORD *)i = v15;
        *(_QWORD *)(i + 8) = i + 24;
        if ( *((_DWORD *)v13 + 4) )
        {
          v26 = v3;
          v28 = v13;
          sub_1BC14B0(i + 8, (__int64)(v13 + 1));
          v3 = v26;
          v13 = v28;
        }
      }
      v13 += 27;
      if ( v3 == v13 )
        break;
    }
    v11 = i + 432;
  }
  if ( v3 != (__int64 *)v5 )
  {
    do
    {
      v16 = *v4;
      *(_DWORD *)(v11 + 16) = 0;
      *(_DWORD *)(v11 + 20) = 8;
      *(_QWORD *)v11 = v16;
      *(_QWORD *)(v11 + 8) = v11 + 24;
      if ( *((_DWORD *)v4 + 4) )
        sub_1BC14B0(v11 + 8, (__int64)(v4 + 1));
      v4 += 27;
      v11 += 216;
    }
    while ( (__int64 *)v5 != v4 );
  }
  for ( j = v32; j != (__int64 *)v5; j += 27 )
  {
    v18 = (__int64 *)j[1];
    v19 = &v18[3 * *((unsigned int *)j + 4)];
    if ( v18 != v19 )
    {
      do
      {
        v20 = *(v19 - 1);
        v19 -= 3;
        if ( v20 != -8 && v20 != 0 && v20 != -16 )
          sub_1649B30(v19);
      }
      while ( v18 != v19 );
      v19 = (__int64 *)j[1];
    }
    if ( v19 != j + 3 )
      _libc_free((unsigned __int64)v19);
  }
  if ( v32 )
    j_j___libc_free_0(v32, a1[2] - (_QWORD)v32);
  *a1 = v31;
  a1[1] = v11;
  a1[2] = v25;
  return a1;
}
