// Function: sub_FE67B0
// Address: 0xfe67b0
//
__int64 __fastcall sub_FE67B0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 result; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // r13
  _QWORD *i; // rbx
  _QWORD *v12; // r14
  _QWORD *v13; // r13
  __int64 v14; // rdi
  __int64 *v15; // rbx
  unsigned __int64 v16; // r12
  __int64 v17; // rdi
  _QWORD *v18; // r14
  __int64 v19; // rdi
  __int64 *v20; // rbx
  unsigned __int64 v21; // r12
  __int64 v22; // rdi
  _QWORD *v23; // [rsp+0h] [rbp-B0h] BYREF
  int v24; // [rsp+8h] [rbp-A8h]
  __int64 v25; // [rsp+10h] [rbp-A0h]
  _QWORD *v26; // [rsp+18h] [rbp-98h]
  _QWORD *v27; // [rsp+20h] [rbp-90h]
  __int64 v28; // [rsp+28h] [rbp-88h]
  __int64 v29; // [rsp+30h] [rbp-80h]
  __int64 v30; // [rsp+38h] [rbp-78h]
  __int64 v31; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v32; // [rsp+48h] [rbp-68h]
  char v33; // [rsp+80h] [rbp-30h] BYREF

  v3 = &v31;
  v23 = a1;
  v24 = -1;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 1;
  do
  {
    *(_DWORD *)v3 = -1;
    v3 += 2;
  }
  while ( v3 != (__int64 *)&v33 );
  sub_FE6400((__int64)&v23, a2, (__int64)a1);
  result = ((__int64 (__fastcall *)(_QWORD *, _QWORD **, __int64, __int64))sub_FEC920)(a1, &v23, a2, a3);
  v10 = v6;
  for ( i = (_QWORD *)result; i != v10; i = (_QWORD *)*i )
    result = sub_FE1320(a1, (__int64)(i + 2));
  if ( a2 )
  {
    result = sub_FE8B10(a1, a2, v6, v7, v8, v9, v23, v24, v25);
    if ( (v30 & 1) == 0 )
      result = sub_C7D6A0(v31, 16LL * v32, 8);
    v12 = v27;
    v13 = v26;
    if ( v27 != v26 )
    {
      do
      {
        v14 = v13[1];
        if ( v14 )
        {
          v15 = (__int64 *)v13[6];
          v16 = v13[10] + 8LL;
          if ( v16 > (unsigned __int64)v15 )
          {
            do
            {
              v17 = *v15++;
              j_j___libc_free_0(v17, 512);
            }
            while ( v16 > (unsigned __int64)v15 );
            v14 = v13[1];
          }
          result = j_j___libc_free_0(v14, 8LL * v13[2]);
        }
        v13 += 11;
      }
      while ( v12 != v13 );
LABEL_15:
      v13 = v26;
    }
  }
  else
  {
    if ( (v30 & 1) == 0 )
      result = sub_C7D6A0(v31, 16LL * v32, 8);
    v18 = v27;
    v13 = v26;
    if ( v27 != v26 )
    {
      do
      {
        v19 = v13[1];
        if ( v19 )
        {
          v20 = (__int64 *)v13[6];
          v21 = v13[10] + 8LL;
          if ( v21 > (unsigned __int64)v20 )
          {
            do
            {
              v22 = *v20++;
              j_j___libc_free_0(v22, 512);
            }
            while ( v21 > (unsigned __int64)v20 );
            v19 = v13[1];
          }
          result = j_j___libc_free_0(v19, 8LL * v13[2]);
        }
        v13 += 11;
      }
      while ( v18 != v13 );
      goto LABEL_15;
    }
  }
  if ( v13 )
    return j_j___libc_free_0(v13, v28 - (_QWORD)v13);
  return result;
}
