// Function: sub_187D570
// Address: 0x187d570
//
__int64 __fastcall sub_187D570(_QWORD *a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r13
  __int64 v4; // r15
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi

  v2 = (_QWORD *)a1[1];
  v3 = (_QWORD *)*a1;
  if ( v2 != (_QWORD *)*a1 )
  {
    do
    {
      v4 = v3[17];
      v5 = v3[16];
      if ( v4 != v5 )
      {
        do
        {
          v6 = *(_QWORD *)(v5 + 16);
          if ( v6 )
            result = j_j___libc_free_0(v6, *(_QWORD *)(v5 + 32) - v6);
          v5 += 40;
        }
        while ( v4 != v5 );
        v5 = v3[16];
      }
      if ( v5 )
        result = j_j___libc_free_0(v5, v3[18] - v5);
      v8 = v3[14];
      v9 = v3[13];
      if ( v8 != v9 )
      {
        do
        {
          v10 = *(_QWORD *)(v9 + 16);
          if ( v10 )
            result = j_j___libc_free_0(v10, *(_QWORD *)(v9 + 32) - v10);
          v9 += 40;
        }
        while ( v8 != v9 );
        v9 = v3[13];
      }
      if ( v9 )
        result = j_j___libc_free_0(v9, v3[15] - v9);
      v11 = v3[10];
      if ( v11 )
        result = j_j___libc_free_0(v11, v3[12] - v11);
      v12 = v3[7];
      if ( v12 )
        result = j_j___libc_free_0(v12, v3[9] - v12);
      v13 = v3[4];
      if ( v13 )
        result = j_j___libc_free_0(v13, v3[6] - v13);
      v14 = v3[1];
      if ( v14 )
        result = j_j___libc_free_0(v14, v3[3] - v14);
      v3 += 19;
    }
    while ( v2 != v3 );
    v3 = (_QWORD *)*a1;
  }
  if ( v3 )
    return j_j___libc_free_0(v3, a1[2] - (_QWORD)v3);
  return result;
}
