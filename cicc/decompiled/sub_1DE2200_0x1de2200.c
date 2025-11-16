// Function: sub_1DE2200
// Address: 0x1de2200
//
__int64 __fastcall sub_1DE2200(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 result; // rax
  _QWORD *v6; // rdx
  _QWORD *v7; // r13
  _QWORD *i; // rbx
  _QWORD *v9; // r14
  _QWORD *v10; // r13
  __int64 v11; // rdi
  __int64 *v12; // rbx
  unsigned __int64 v13; // r12
  __int64 v14; // rdi
  _QWORD *v15; // r14
  __int64 v16; // rdi
  __int64 *v17; // rbx
  unsigned __int64 v18; // r12
  __int64 v19; // rdi
  __int64 v20; // [rsp+0h] [rbp-B0h] BYREF
  int v21; // [rsp+8h] [rbp-A8h]
  __int64 v22; // [rsp+10h] [rbp-A0h]
  _QWORD *v23; // [rsp+18h] [rbp-98h]
  _QWORD *v24; // [rsp+20h] [rbp-90h]
  __int64 v25; // [rsp+28h] [rbp-88h]
  __int64 v26; // [rsp+30h] [rbp-80h]
  __int64 v27; // [rsp+38h] [rbp-78h]
  __int64 v28; // [rsp+40h] [rbp-70h] BYREF
  char v29; // [rsp+80h] [rbp-30h] BYREF

  v3 = &v28;
  v20 = a1;
  v21 = -1;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 1;
  do
  {
    *(_DWORD *)v3 = -1;
    v3 += 2;
  }
  while ( v3 != (__int64 *)&v29 );
  sub_1DE1E40((__int64)&v20, a2, a1);
  result = sub_13751A0(a1, (__int64)&v20, a2, a3);
  v7 = v6;
  for ( i = (_QWORD *)result; i != v7; i = (_QWORD *)*i )
    result = sub_1DDED90(a1, (__int64)(i + 2));
  if ( a2 )
  {
    result = sub_1371020(a1, a2);
    if ( (v27 & 1) == 0 )
      result = j___libc_free_0(v28);
    v9 = v24;
    v10 = v23;
    if ( v24 != v23 )
    {
      do
      {
        v11 = v10[1];
        if ( v11 )
        {
          v12 = (__int64 *)v10[6];
          v13 = v10[10] + 8LL;
          if ( v13 > (unsigned __int64)v12 )
          {
            do
            {
              v14 = *v12++;
              j_j___libc_free_0(v14, 512);
            }
            while ( v13 > (unsigned __int64)v12 );
            v11 = v10[1];
          }
          result = j_j___libc_free_0(v11, 8LL * v10[2]);
        }
        v10 += 11;
      }
      while ( v9 != v10 );
LABEL_15:
      v10 = v23;
    }
  }
  else
  {
    if ( (v27 & 1) == 0 )
      result = j___libc_free_0(v28);
    v15 = v24;
    v10 = v23;
    if ( v24 != v23 )
    {
      do
      {
        v16 = v10[1];
        if ( v16 )
        {
          v17 = (__int64 *)v10[6];
          v18 = v10[10] + 8LL;
          if ( v18 > (unsigned __int64)v17 )
          {
            do
            {
              v19 = *v17++;
              j_j___libc_free_0(v19, 512);
            }
            while ( v18 > (unsigned __int64)v17 );
            v16 = v10[1];
          }
          result = j_j___libc_free_0(v16, 8LL * v10[2]);
        }
        v10 += 11;
      }
      while ( v15 != v10 );
      goto LABEL_15;
    }
  }
  if ( v10 )
    return j_j___libc_free_0(v10, v25 - (_QWORD)v10);
  return result;
}
