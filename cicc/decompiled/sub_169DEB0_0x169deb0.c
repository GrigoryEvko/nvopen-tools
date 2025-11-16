// Function: sub_169DEB0
// Address: 0x169deb0
//
void __fastcall sub_169DEB0(__int64 *a1)
{
  __int64 v1; // r14
  __int64 v2; // rsi
  __int64 v3; // rbx
  void *v4; // r13
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-40h]
  __int64 v13; // [rsp+8h] [rbp-38h]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v1 = *a1;
  if ( *a1 )
  {
    v2 = 32LL * *(_QWORD *)(v1 - 8);
    v3 = v1 + v2;
    if ( v1 == v1 + v2 )
    {
LABEL_21:
      j_j_j___libc_free_0_0(v1 - 8);
      return;
    }
    v4 = sub_16982C0();
    while ( 1 )
    {
      while ( 1 )
      {
        v3 -= 32;
        if ( *(void **)(v3 + 8) == v4 )
          break;
        sub_1698460(v3 + 8);
LABEL_5:
        if ( v1 == v3 )
          goto LABEL_21;
      }
      v5 = *(_QWORD *)(v3 + 16);
      if ( !v5 )
        goto LABEL_5;
      v6 = 32LL * *(_QWORD *)(v5 - 8);
      v7 = v5 + v6;
      while ( v5 != v7 )
      {
        while ( 1 )
        {
          v7 -= 32;
          if ( v4 == *(void **)(v7 + 8) )
            break;
          sub_1698460(v7 + 8);
LABEL_11:
          if ( v5 == v7 )
            goto LABEL_20;
        }
        v8 = *(_QWORD *)(v7 + 16);
        if ( !v8 )
          goto LABEL_11;
        v9 = 32LL * *(_QWORD *)(v8 - 8);
        v10 = v8 + v9;
        if ( v8 != v8 + v9 )
        {
          do
          {
            while ( 1 )
            {
              v11 = v10 - 32;
              v12 = v8;
              if ( v4 == *(void **)(v11 + 8) )
                break;
              v13 = v11;
              sub_1698460(v11 + 8);
              v10 = v13;
              v8 = v12;
              if ( v12 == v13 )
                goto LABEL_19;
            }
            v14 = v11;
            sub_169DEB0(v11 + 16);
            v8 = v12;
            v10 = v14;
          }
          while ( v12 != v14 );
        }
LABEL_19:
        j_j_j___libc_free_0_0(v8 - 8);
      }
LABEL_20:
      j_j_j___libc_free_0_0(v5 - 8);
      if ( v1 == v3 )
        goto LABEL_21;
    }
  }
}
