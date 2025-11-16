// Function: sub_16A0170
// Address: 0x16a0170
//
__int64 *__fastcall sub_16A0170(__int64 *a1, __int64 *a2)
{
  __int64 *v3; // r12
  __int64 v4; // r13
  __int64 v5; // rsi
  __int64 v6; // rbx
  void *v7; // r15
  __int64 *v8; // rsi
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 *v12; // rdi
  __int64 *v13; // rsi
  void *v14; // r13
  void *v15; // rax
  void *v16; // rax
  __int64 *v17; // rsi
  __int64 *v18; // [rsp+0h] [rbp-40h]
  __int64 *v19; // [rsp+8h] [rbp-38h]

  v3 = a1;
  if ( *a1 == *a2 )
  {
    v10 = a2[1];
    if ( v10 )
    {
      v11 = a1[1];
      v18 = (__int64 *)(v10 + 8);
      v19 = (__int64 *)(v11 + 8);
      v12 = (__int64 *)(v11 + 8);
      v13 = (__int64 *)(v10 + 8);
      v14 = sub_16982C0();
      v15 = *(void **)(v10 + 8);
      if ( *(void **)(v11 + 8) == v14 )
      {
        if ( v14 == v15 )
        {
          sub_16A0170(v12, v13);
          v10 = a2[1];
          v11 = v3[1];
          goto LABEL_17;
        }
      }
      else if ( v14 != v15 )
      {
        sub_1698680(v12, v13);
        v10 = a2[1];
        v11 = v3[1];
LABEL_17:
        v16 = *(void **)(v10 + 40);
        v17 = (__int64 *)(v10 + 40);
        if ( v14 == *(void **)(v11 + 40) )
        {
          if ( v14 == v16 )
          {
            sub_16A0170(v11 + 40, v17);
            return v3;
          }
        }
        else if ( v14 != v16 )
        {
          sub_1698680((__int64 *)(v11 + 40), v17);
          return v3;
        }
        if ( v17 == (__int64 *)(v11 + 40) )
          return v3;
        sub_127D120((_QWORD *)(v11 + 40));
        v8 = (__int64 *)(v10 + 40);
        a1 = (__int64 *)(v11 + 40);
        if ( v14 != *(void **)(v10 + 40) )
        {
          sub_16986C0(a1, v8);
          return v3;
        }
        goto LABEL_11;
      }
      if ( v18 != v19 )
      {
        sub_127D120(v12);
        if ( v14 == *(void **)(v10 + 8) )
          sub_169C6E0(v19, (__int64)v18);
        else
          sub_16986C0(v19, v18);
        v10 = a2[1];
        v11 = v3[1];
      }
      goto LABEL_17;
    }
  }
  if ( a1 != a2 )
  {
    v4 = a1[1];
    if ( v4 )
    {
      v5 = 32LL * *(_QWORD *)(v4 - 8);
      v6 = v4 + v5;
      if ( v4 != v4 + v5 )
      {
        v7 = sub_16982C0();
        do
        {
          while ( 1 )
          {
            v6 -= 32;
            if ( *(void **)(v6 + 8) == v7 )
              break;
            sub_1698460(v6 + 8);
            if ( v4 == v6 )
              goto LABEL_9;
          }
          sub_169DEB0((__int64 *)(v6 + 16));
        }
        while ( v4 != v6 );
      }
LABEL_9:
      j_j_j___libc_free_0_0(v4 - 8);
    }
    v8 = a2;
LABEL_11:
    sub_169C6E0(a1, (__int64)v8);
  }
  return v3;
}
