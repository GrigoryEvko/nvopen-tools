// Function: sub_2627CA0
// Address: 0x2627ca0
//
void __fastcall sub_2627CA0(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r13
  __int64 v4; // r15
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  __int64 v7; // r15
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi

  v2 = *(_QWORD **)(a1 + 8);
  v3 = *(_QWORD **)a1;
  if ( v2 != *(_QWORD **)a1 )
  {
    do
    {
      v4 = v3[20];
      v5 = v3[19];
      if ( v4 != v5 )
      {
        do
        {
          v6 = *(_QWORD *)(v5 + 16);
          if ( v6 )
            j_j___libc_free_0(v6);
          v5 += 40LL;
        }
        while ( v4 != v5 );
        v5 = v3[19];
      }
      if ( v5 )
        j_j___libc_free_0(v5);
      v7 = v3[17];
      v8 = v3[16];
      if ( v7 != v8 )
      {
        do
        {
          v9 = *(_QWORD *)(v8 + 16);
          if ( v9 )
            j_j___libc_free_0(v9);
          v8 += 40LL;
        }
        while ( v7 != v8 );
        v8 = v3[16];
      }
      if ( v8 )
        j_j___libc_free_0(v8);
      v10 = v3[13];
      if ( v10 )
        j_j___libc_free_0(v10);
      v11 = v3[10];
      if ( v11 )
        j_j___libc_free_0(v11);
      v12 = v3[7];
      if ( v12 )
        j_j___libc_free_0(v12);
      v13 = v3[4];
      if ( v13 )
        j_j___libc_free_0(v13);
      v3 += 22;
    }
    while ( v2 != v3 );
    v3 = *(_QWORD **)a1;
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
}
