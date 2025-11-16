// Function: sub_E63CD0
// Address: 0xe63cd0
//
__int64 __fastcall sub_E63CD0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r13
  _QWORD *v4; // r14
  _QWORD *v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // rdi
  _QWORD *v9; // rdi
  __int64 v10; // r8
  __int64 v11; // r14
  __int64 v12; // r14
  __int64 v13; // r12
  _QWORD *v14; // rdi
  _QWORD *v15; // r14
  _QWORD *v16; // r12
  _QWORD *v17; // r14
  _QWORD *v18; // r12
  __int64 result; // rax

  if ( a1 )
  {
    v2 = a1;
    do
    {
      v3 = v2;
      sub_E63CD0(v2[3]);
      v4 = (_QWORD *)v2[74];
      v2 = (_QWORD *)v2[2];
      v5 = &v4[4 * *((unsigned int *)v3 + 150)];
      if ( v4 != v5 )
      {
        do
        {
          v6 = *(v5 - 3);
          v5 -= 4;
          if ( v6 )
          {
            a2 = v5[3] - v6;
            j_j___libc_free_0(v6, a2);
          }
        }
        while ( v4 != v5 );
        v5 = (_QWORD *)v3[74];
      }
      if ( v5 != v3 + 76 )
        _libc_free(v5, a2);
      v7 = 16LL * *((unsigned int *)v3 + 146);
      sub_C7D6A0(v3[71], v7, 8);
      v8 = (_QWORD *)v3[59];
      if ( v8 != v3 + 61 )
      {
        v7 = v3[61] + 1LL;
        j_j___libc_free_0(v8, v7);
      }
      v9 = (_QWORD *)v3[55];
      if ( v9 != v3 + 57 )
      {
        v7 = v3[57] + 1LL;
        j_j___libc_free_0(v9, v7);
      }
      v10 = v3[52];
      if ( *((_DWORD *)v3 + 107) )
      {
        v11 = *((unsigned int *)v3 + 106);
        if ( (_DWORD)v11 )
        {
          v12 = 8 * v11;
          v13 = 0;
          do
          {
            v14 = *(_QWORD **)(v10 + v13);
            if ( v14 != (_QWORD *)-8LL && v14 )
            {
              v7 = *v14 + 17LL;
              sub_C7D6A0((__int64)v14, v7, 8);
              v10 = v3[52];
            }
            v13 += 8;
          }
          while ( v12 != v13 );
        }
      }
      _libc_free(v10, v7);
      v15 = (_QWORD *)v3[20];
      v16 = &v15[10 * *((unsigned int *)v3 + 42)];
      if ( v15 != v16 )
      {
        do
        {
          v16 -= 10;
          if ( (_QWORD *)*v16 != v16 + 2 )
          {
            v7 = v16[2] + 1LL;
            j_j___libc_free_0(*v16, v7);
          }
        }
        while ( v15 != v16 );
        v16 = (_QWORD *)v3[20];
      }
      if ( v16 != v3 + 22 )
        _libc_free(v16, v7);
      v17 = (_QWORD *)v3[6];
      v18 = &v17[4 * *((unsigned int *)v3 + 14)];
      if ( v17 != v18 )
      {
        do
        {
          v18 -= 4;
          if ( (_QWORD *)*v18 != v18 + 2 )
          {
            v7 = v18[2] + 1LL;
            j_j___libc_free_0(*v18, v7);
          }
        }
        while ( v17 != v18 );
        v18 = (_QWORD *)v3[6];
      }
      if ( v18 != v3 + 8 )
        _libc_free(v18, v7);
      a2 = 608;
      result = j_j___libc_free_0(v3, 608);
    }
    while ( v2 );
  }
  return result;
}
