// Function: sub_26FD970
// Address: 0x26fd970
//
void __fastcall sub_26FD970(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r8
  _BYTE *v3; // rsi
  unsigned __int64 *v5; // r13
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  _QWORD v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD **)(a1 + 80);
  v14[0] = a2;
  if ( !v2 )
  {
    v2 = (_QWORD *)sub_22077B0(0x78u);
    if ( v2 )
      memset(v2, 0, 0x78u);
    v5 = *(unsigned __int64 **)(a1 + 80);
    *(_QWORD *)(a1 + 80) = v2;
    if ( v5 )
    {
      v6 = v5[13];
      v7 = v5[12];
      if ( v6 != v7 )
      {
        do
        {
          v8 = *(_QWORD *)(v7 + 16);
          if ( v8 )
            j_j___libc_free_0(v8);
          v7 += 40LL;
        }
        while ( v6 != v7 );
        v7 = v5[12];
      }
      if ( v7 )
        j_j___libc_free_0(v7);
      v9 = v5[10];
      v10 = v5[9];
      if ( v9 != v10 )
      {
        do
        {
          v11 = *(_QWORD *)(v10 + 16);
          if ( v11 )
            j_j___libc_free_0(v11);
          v10 += 40LL;
        }
        while ( v9 != v10 );
        v10 = v5[9];
      }
      if ( v10 )
        j_j___libc_free_0(v10);
      v12 = v5[6];
      if ( v12 )
        j_j___libc_free_0(v12);
      v13 = v5[3];
      if ( v13 )
        j_j___libc_free_0(v13);
      if ( *v5 )
        j_j___libc_free_0(*v5);
      j_j___libc_free_0((unsigned __int64)v5);
      v2 = *(_QWORD **)(a1 + 80);
    }
  }
  v3 = (_BYTE *)v2[1];
  if ( v3 == (_BYTE *)v2[2] )
  {
    sub_9CA200((__int64)v2, v3, v14);
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = v14[0];
      v3 = (_BYTE *)v2[1];
    }
    v2[1] = v3 + 8;
  }
}
