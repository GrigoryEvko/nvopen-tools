// Function: sub_31142D0
// Address: 0x31142d0
//
void __fastcall sub_31142D0(unsigned __int64 a1)
{
  unsigned __int64 v1; // r14
  _QWORD *v2; // rbx
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r15
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v10; // [rsp+8h] [rbp-48h]
  _QWORD *v11; // [rsp+10h] [rbp-40h]
  _QWORD *v12; // [rsp+18h] [rbp-38h]

  v11 = *(_QWORD **)(a1 + 32);
  while ( v11 )
  {
    v10 = (unsigned __int64)v11;
    v1 = v11[2];
    v11 = (_QWORD *)*v11;
    if ( v1 )
    {
      v2 = *(_QWORD **)(v1 + 32);
      while ( v2 )
      {
        v3 = (unsigned __int64)v2;
        v2 = (_QWORD *)*v2;
        v4 = *(_QWORD *)(v3 + 16);
        if ( v4 )
        {
          v5 = *(_QWORD **)(v4 + 32);
          if ( v5 )
          {
            while ( 1 )
            {
              v12 = (_QWORD *)*v5;
              if ( v5[2] )
                sub_31142D0();
              j_j___libc_free_0((unsigned __int64)v5);
              if ( !v12 )
                break;
              v5 = v12;
            }
          }
          memset(*(void **)(v4 + 16), 0, 8LL * *(_QWORD *)(v4 + 24));
          v6 = *(_QWORD *)(v4 + 16);
          *(_QWORD *)(v4 + 40) = 0;
          *(_QWORD *)(v4 + 32) = 0;
          if ( v6 != v4 + 64 )
            j_j___libc_free_0(v6);
          j_j___libc_free_0(v4);
        }
        j_j___libc_free_0(v3);
      }
      memset(*(void **)(v1 + 16), 0, 8LL * *(_QWORD *)(v1 + 24));
      v7 = *(_QWORD *)(v1 + 16);
      *(_QWORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 32) = 0;
      if ( v7 != v1 + 64 )
        j_j___libc_free_0(v7);
      j_j___libc_free_0(v1);
    }
    j_j___libc_free_0(v10);
  }
  memset(*(void **)(a1 + 16), 0, 8LL * *(_QWORD *)(a1 + 24));
  v8 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  if ( v8 != a1 + 64 )
    j_j___libc_free_0(v8);
  j_j___libc_free_0(a1);
}
