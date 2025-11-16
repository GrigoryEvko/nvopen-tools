// Function: sub_1549650
// Address: 0x1549650
//
__int64 __fastcall sub_1549650(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // r13
  unsigned __int64 v13; // r8
  __int64 v14; // r13
  __int64 v15; // rbx
  unsigned __int64 v16; // rdi

  j___libc_free_0(a1[79]);
  v2 = a1[59];
  if ( (_QWORD *)v2 != a1 + 61 )
    _libc_free(v2);
  v3 = a1[41];
  if ( (_QWORD *)v3 != a1 + 43 )
    _libc_free(v3);
  v4 = a1[39];
  v5 = a1[38];
  if ( v4 != v5 )
  {
    do
    {
      v6 = *(_QWORD *)(v5 + 16);
      if ( v6 )
        j_j___libc_free_0(v6, *(_QWORD *)(v5 + 32) - v6);
      v5 += 40;
    }
    while ( v4 != v5 );
    v5 = a1[38];
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[40] - v5);
  v7 = a1[34];
  if ( v7 )
    j_j___libc_free_0(v7, a1[36] - v7);
  j___libc_free_0(a1[31]);
  v8 = a1[26];
  if ( v8 )
    j_j___libc_free_0(v8, a1[28] - v8);
  j___libc_free_0(a1[23]);
  v9 = a1[18];
  if ( v9 )
    j_j___libc_free_0(v9, a1[20] - v9);
  j___libc_free_0(a1[15]);
  j___libc_free_0(a1[11]);
  result = j___libc_free_0(a1[7]);
  v11 = a1[3];
  if ( v11 )
  {
    j___libc_free_0(*(_QWORD *)(v11 + 240));
    if ( *(_DWORD *)(v11 + 204) )
    {
      v12 = *(unsigned int *)(v11 + 200);
      v13 = *(_QWORD *)(v11 + 192);
      if ( (_DWORD)v12 )
      {
        v14 = 8 * v12;
        v15 = 0;
        do
        {
          v16 = *(_QWORD *)(v13 + v15);
          if ( v16 != -8 )
          {
            if ( v16 )
            {
              _libc_free(v16);
              v13 = *(_QWORD *)(v11 + 192);
            }
          }
          v15 += 8;
        }
        while ( v14 != v15 );
      }
    }
    else
    {
      v13 = *(_QWORD *)(v11 + 192);
    }
    _libc_free(v13);
    j___libc_free_0(*(_QWORD *)(v11 + 160));
    j___libc_free_0(*(_QWORD *)(v11 + 120));
    j___libc_free_0(*(_QWORD *)(v11 + 80));
    j___libc_free_0(*(_QWORD *)(v11 + 40));
    return j_j___libc_free_0(v11, 272);
  }
  return result;
}
