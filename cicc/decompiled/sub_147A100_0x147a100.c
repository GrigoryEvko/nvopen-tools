// Function: sub_147A100
// Address: 0x147a100
//
__int64 __fastcall sub_147A100(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 *v7; // r15
  __int64 *v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // rdx
  unsigned int v15; // r12d
  __int64 *i; // rcx
  __int64 *v18; // r14
  __int64 j; // rax
  __int64 v22; // [rsp+18h] [rbp-A8h]
  __int64 v23; // [rsp+20h] [rbp-A0h] BYREF
  __int64 *v24; // [rsp+28h] [rbp-98h]
  __int64 *v25; // [rsp+30h] [rbp-90h]
  __int64 v26; // [rsp+38h] [rbp-88h]
  int v27; // [rsp+40h] [rbp-80h]
  _BYTE v28[120]; // [rsp+48h] [rbp-78h] BYREF

  v24 = (__int64 *)v28;
  v25 = (__int64 *)v28;
  v23 = 0;
  v26 = 8;
  v27 = 0;
  sub_145B270(a1, a3, (__int64)&v23);
  sub_145B270(a1, a4, (__int64)&v23);
  v6 = HIDWORD(v26);
  if ( v27 == HIDWORD(v26) )
    goto LABEL_13;
  v7 = v25;
  if ( v25 != v24 )
    v6 = (unsigned int)v26;
  v8 = &v25[v6];
  v9 = *v25;
  if ( v25 != v8 )
  {
    while ( 1 )
    {
      v9 = *v7;
      if ( (unsigned __int64)*v7 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v7 + 1 == v8 )
      {
        v9 = v7[1];
        goto LABEL_10;
      }
      ++v7;
    }
    if ( v7 != v8 )
    {
LABEL_18:
      for ( i = v7; ; i = v18 )
      {
        v18 = i + 1;
        if ( i + 1 == v8 )
          break;
        for ( j = *v18; (unsigned __int64)*v18 >= 0xFFFFFFFFFFFFFFFELL; j = *v18 )
        {
          if ( v8 == ++v18 )
            goto LABEL_10;
        }
        if ( v18 == v8 )
          break;
        if ( (unsigned __int8)sub_15CC890(*(_QWORD *)(a1 + 56), **(_QWORD **)(v9 + 32), **(_QWORD **)(j + 32), i, v5) )
        {
          v9 = *v18;
          v7 = v18;
          goto LABEL_18;
        }
        v9 = *v7;
      }
    }
  }
LABEL_10:
  v10 = sub_147A000(a1, v9, a3);
  v12 = v11;
  if ( sub_1456E90(a1) == v10
    || (v13 = sub_147A000(a1, v9, a4), v22 = v14, sub_1456E90(a1) == v13)
    || !sub_146D950(a1, v10, v9)
    || !sub_146D950(a1, v13, v9)
    || !(unsigned __int8)sub_148B410(a1, v9, a2, v10, v13) )
  {
LABEL_13:
    v15 = 0;
  }
  else
  {
    v15 = sub_1474350(a1, v9, a2, v12, v22);
  }
  if ( v25 != v24 )
    _libc_free((unsigned __int64)v25);
  return v15;
}
