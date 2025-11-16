// Function: sub_2E42D20
// Address: 0x2e42d20
//
void __fastcall sub_2E42D20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v5; // rax
  _QWORD *v6; // rdx
  _QWORD *v7; // r13
  _QWORD *i; // rbx
  _QWORD *v9; // r14
  _QWORD *v10; // r13
  unsigned __int64 v11; // rdi
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rdi
  _QWORD *v15; // r14
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // rbx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // [rsp+0h] [rbp-B0h] BYREF
  int v21; // [rsp+8h] [rbp-A8h]
  __int64 v22; // [rsp+10h] [rbp-A0h]
  _QWORD *v23; // [rsp+18h] [rbp-98h]
  _QWORD *v24; // [rsp+20h] [rbp-90h]
  __int64 v25; // [rsp+28h] [rbp-88h]
  __int64 v26; // [rsp+30h] [rbp-80h]
  __int64 v27; // [rsp+38h] [rbp-78h]
  __int64 v28; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v29; // [rsp+48h] [rbp-68h]
  char v30; // [rsp+80h] [rbp-30h] BYREF

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
  while ( v3 != (__int64 *)&v30 );
  sub_2E42970((__int64)&v20, a2, a1);
  v5 = sub_FEC920(a1, (__int64)&v20, a2, a3);
  v7 = v6;
  for ( i = (_QWORD *)v5; i != v7; i = (_QWORD *)*i )
    sub_2E3DDD0(a1, (__int64)(i + 2));
  if ( a2 )
  {
    sub_FE8B10(a1, a2);
    if ( (v27 & 1) == 0 )
      sub_C7D6A0(v28, 16LL * v29, 8);
    v9 = v24;
    v10 = v23;
    if ( v24 != v23 )
    {
      do
      {
        v11 = v10[1];
        if ( v11 )
        {
          v12 = (unsigned __int64 *)v10[6];
          v13 = v10[10] + 8LL;
          if ( v13 > (unsigned __int64)v12 )
          {
            do
            {
              v14 = *v12++;
              j_j___libc_free_0(v14);
            }
            while ( v13 > (unsigned __int64)v12 );
            v11 = v10[1];
          }
          j_j___libc_free_0(v11);
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
      sub_C7D6A0(v28, 16LL * v29, 8);
    v15 = v24;
    v10 = v23;
    if ( v24 != v23 )
    {
      do
      {
        v16 = v10[1];
        if ( v16 )
        {
          v17 = (unsigned __int64 *)v10[6];
          v18 = v10[10] + 8LL;
          if ( v18 > (unsigned __int64)v17 )
          {
            do
            {
              v19 = *v17++;
              j_j___libc_free_0(v19);
            }
            while ( v18 > (unsigned __int64)v17 );
            v16 = v10[1];
          }
          j_j___libc_free_0(v16);
        }
        v10 += 11;
      }
      while ( v15 != v10 );
      goto LABEL_15;
    }
  }
  if ( v10 )
    j_j___libc_free_0((unsigned __int64)v10);
}
