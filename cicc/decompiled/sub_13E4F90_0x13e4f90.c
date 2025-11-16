// Function: sub_13E4F90
// Address: 0x13e4f90
//
__int64 __fastcall sub_13E4F90(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  _QWORD *v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 result; // rax

  v2 = a1[27];
  v3 = (a1[28] - v2) >> 3;
  if ( (_DWORD)v3 )
  {
    v4 = 0;
    v5 = 8LL * (unsigned int)(v3 - 1);
    while ( 1 )
    {
      v6 = *(_QWORD **)(v2 + v4);
      if ( v6 )
      {
        v7 = v6[7];
        if ( v7 )
          j_j___libc_free_0(v7, v6[9] - v7);
        v8 = v6[4];
        if ( v8 )
          j_j___libc_free_0(v8, v6[6] - v8);
        v9 = v6[1];
        if ( v9 )
          j_j___libc_free_0(v9, v6[3] - v9);
        j_j___libc_free_0(v6, 80);
      }
      if ( v4 == v5 )
        break;
      v2 = a1[27];
      v4 += 8;
    }
  }
  v10 = a1[22];
  while ( v10 )
  {
    sub_13E4CB0(*(_QWORD *)(v10 + 24));
    v11 = v10;
    v10 = *(_QWORD *)(v10 + 16);
    j_j___libc_free_0(v11, 48);
  }
  a1[22] = 0;
  a1[23] = a1 + 21;
  a1[24] = a1 + 21;
  result = a1[27];
  a1[25] = 0;
  if ( result != a1[28] )
    a1[28] = result;
  a1[26] = 0;
  return result;
}
