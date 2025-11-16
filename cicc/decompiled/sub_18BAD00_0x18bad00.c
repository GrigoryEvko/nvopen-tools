// Function: sub_18BAD00
// Address: 0x18bad00
//
char *__fastcall sub_18BAD00(__int64 a1, char *a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rsi
  char *result; // rax
  _QWORD *v6; // r13
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  _QWORD *v15; // r13
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // rdi
  __int64 v19; // r14
  __int64 v20; // r12
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  char *v24; // [rsp+8h] [rbp-38h] BYREF
  _QWORD *v25; // [rsp+18h] [rbp-28h] BYREF

  v3 = *(_QWORD *)(a1 + 96);
  v24 = a2;
  if ( !v3 )
  {
    result = (char *)sub_18B9090(&v25);
    v3 = (__int64)v25;
    v6 = *(_QWORD **)(a1 + 96);
    v25 = 0;
    *(_QWORD *)(a1 + 96) = v3;
    if ( v6 )
    {
      v7 = v6[13];
      v8 = v6[12];
      if ( v7 != v8 )
      {
        do
        {
          v9 = *(_QWORD *)(v8 + 16);
          if ( v9 )
            j_j___libc_free_0(v9, *(_QWORD *)(v8 + 32) - v9);
          v8 += 40;
        }
        while ( v7 != v8 );
        v8 = v6[12];
      }
      if ( v8 )
        j_j___libc_free_0(v8, v6[14] - v8);
      v10 = v6[10];
      v11 = v6[9];
      if ( v10 != v11 )
      {
        do
        {
          v12 = *(_QWORD *)(v11 + 16);
          if ( v12 )
            j_j___libc_free_0(v12, *(_QWORD *)(v11 + 32) - v12);
          v11 += 40;
        }
        while ( v10 != v11 );
        v11 = v6[9];
      }
      if ( v11 )
        j_j___libc_free_0(v11, v6[11] - v11);
      v13 = v6[6];
      if ( v13 )
        j_j___libc_free_0(v13, v6[8] - v13);
      v14 = v6[3];
      if ( v14 )
        j_j___libc_free_0(v14, v6[5] - v14);
      if ( *v6 )
        j_j___libc_free_0(*v6, v6[2] - *v6);
      result = (char *)j_j___libc_free_0(v6, 120);
      v15 = v25;
      if ( v25 )
      {
        v16 = v25[13];
        v17 = v25[12];
        if ( v16 != v17 )
        {
          do
          {
            v18 = *(_QWORD *)(v17 + 16);
            if ( v18 )
              j_j___libc_free_0(v18, *(_QWORD *)(v17 + 32) - v18);
            v17 += 40;
          }
          while ( v16 != v17 );
          v17 = v15[12];
        }
        if ( v17 )
          j_j___libc_free_0(v17, v15[14] - v17);
        v19 = v15[10];
        v20 = v15[9];
        if ( v19 != v20 )
        {
          do
          {
            v21 = *(_QWORD *)(v20 + 16);
            if ( v21 )
              j_j___libc_free_0(v21, *(_QWORD *)(v20 + 32) - v21);
            v20 += 40;
          }
          while ( v19 != v20 );
          v20 = v15[9];
        }
        if ( v20 )
          j_j___libc_free_0(v20, v15[11] - v20);
        v22 = v15[6];
        if ( v22 )
          j_j___libc_free_0(v22, v15[8] - v22);
        v23 = v15[3];
        if ( v23 )
          j_j___libc_free_0(v23, v15[5] - v23);
        if ( *v15 )
          j_j___libc_free_0(*v15, v15[2] - *v15);
        result = (char *)j_j___libc_free_0(v15, 120);
      }
      v3 = *(_QWORD *)(a1 + 96);
    }
  }
  v4 = *(_BYTE **)(v3 + 8);
  if ( v4 == *(_BYTE **)(v3 + 16) )
    return sub_9CA200(v3, v4, &v24);
  if ( v4 )
  {
    result = v24;
    *(_QWORD *)v4 = v24;
    v4 = *(_BYTE **)(v3 + 8);
  }
  *(_QWORD *)(v3 + 8) = v4 + 8;
  return result;
}
