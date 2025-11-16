// Function: sub_E66210
// Address: 0xe66210
//
_QWORD *__fastcall sub_E66210(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // r12
  _QWORD *v18; // rdi
  _QWORD *v19; // [rsp-30h] [rbp-30h]

  result = *(_QWORD **)(a1 + 184);
  if ( !result )
  {
    result = (_QWORD *)sub_22077B0(312);
    if ( result )
    {
      a2 = (__int64)(result + 8);
      *result = a1;
      result[1] = 0;
      result[3] = 0x1000000000LL;
      result[2] = 0;
      result[4] = 0;
      result[5] = result + 8;
      result[6] = 0;
      result[7] = 0;
      v19 = result;
      sub_C8D290((__int64)(result + 5), result + 8, 1, 1u, v4, v5);
      result = v19;
      *(_BYTE *)(v19[5] + v19[6]) = 0;
      v19[8] = v19 + 10;
      ++v19[6];
      v19[9] = 0x400000000LL;
      *((_DWORD *)v19 + 54) = 0;
      v19[28] = 0;
      v19[29] = v19 + 27;
      v19[30] = v19 + 27;
      v19[31] = 0;
      v19[32] = 0;
      v19[33] = 0;
      v19[34] = 0;
      v19[35] = 0;
      v19[36] = 0;
      v19[37] = 0;
      *((_BYTE *)v19 + 304) = 0;
    }
    v6 = *(_QWORD *)(a1 + 184);
    *(_QWORD *)(a1 + 184) = result;
    if ( v6 )
    {
      v7 = *(_QWORD *)(v6 + 288);
      v8 = *(_QWORD *)(v6 + 280);
      if ( v7 != v8 )
      {
        do
        {
          v9 = *(unsigned int *)(v8 + 48);
          v10 = *(_QWORD *)(v8 + 32);
          v8 += 56;
          a2 = 16 * v9;
          sub_C7D6A0(v10, a2, 4);
        }
        while ( v7 != v8 );
        v8 = *(_QWORD *)(v6 + 280);
      }
      if ( v8 )
      {
        a2 = *(_QWORD *)(v6 + 296) - v8;
        j_j___libc_free_0(v8, a2);
      }
      v11 = *(_QWORD *)(v6 + 256);
      if ( v11 )
      {
        a2 = *(_QWORD *)(v6 + 272) - v11;
        j_j___libc_free_0(v11, a2);
      }
      sub_E62F00(*(_QWORD *)(v6 + 224));
      v12 = *(_QWORD *)(v6 + 64);
      if ( v12 != v6 + 80 )
        _libc_free(v12, a2);
      v13 = *(_QWORD *)(v6 + 40);
      if ( v6 + 64 != v13 )
        _libc_free(v13, a2);
      v14 = *(_QWORD *)(v6 + 8);
      if ( *(_DWORD *)(v6 + 20) )
      {
        v15 = *(unsigned int *)(v6 + 16);
        if ( (_DWORD)v15 )
        {
          v16 = 8 * v15;
          v17 = 0;
          do
          {
            v18 = *(_QWORD **)(v14 + v17);
            if ( v18 != (_QWORD *)-8LL )
            {
              if ( v18 )
              {
                a2 = *v18 + 17LL;
                sub_C7D6A0((__int64)v18, a2, 8);
                v14 = *(_QWORD *)(v6 + 8);
              }
            }
            v17 += 8;
          }
          while ( v16 != v17 );
        }
      }
      _libc_free(v14, a2);
      j_j___libc_free_0(v6, 312);
      return *(_QWORD **)(a1 + 184);
    }
  }
  return result;
}
