// Function: sub_30A78D0
// Address: 0x30a78d0
//
__int64 __fastcall sub_30A78D0(__int64 a1, __int64 a2, __int64 **a3)
{
  __int64 **v3; // r14
  __int64 *v5; // rbx
  __int64 v7; // r12
  size_t v8; // rdx
  const void *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 *v13; // rdi
  __int64 v14; // rax
  size_t v15; // rdx
  __int64 v16; // rcx
  const void *v17; // rsi
  size_t v18; // rax
  __int64 v20; // [rsp+0h] [rbp-60h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  void *v22; // [rsp+8h] [rbp-58h]
  __int64 v23[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF

  v3 = a3 + 3;
  v5 = a3[4];
  if ( v5 != (__int64 *)(a3 + 3) )
  {
    do
    {
      while ( 1 )
      {
        v7 = (__int64)(v5 - 7);
        if ( !v5 )
          v7 = 0;
        if ( !sub_B2FC80(v7) )
        {
          v8 = 0;
          v9 = off_4CE0088;
          if ( off_4CE0088 )
          {
            v9 = off_4CE0088;
            v8 = strlen((const char *)off_4CE0088);
          }
          if ( !sub_B91CC0(v7, v9, v8) )
            break;
        }
        v5 = (__int64 *)v5[1];
        if ( v3 == (__int64 **)v5 )
          goto LABEL_15;
      }
      sub_B2F930(v23, v7);
      v21 = sub_B2F650(v23[0], v23[1]);
      if ( (__int64 *)v23[0] != &v24 )
        j_j___libc_free_0(v23[0]);
      v10 = sub_BCB2E0(*a3);
      v11 = sub_ACD640(v10, v21, 0);
      v12 = sub_B98A20(v11, v21);
      v13 = *a3;
      v23[0] = (__int64)v12;
      v14 = sub_B9C770(v13, v23, (__int64 *)1, 0, 1);
      v15 = 0;
      v16 = v14;
      v17 = off_4CE0088;
      if ( off_4CE0088 )
      {
        v20 = v14;
        v22 = off_4CE0088;
        v18 = strlen((const char *)off_4CE0088);
        v16 = v20;
        v17 = v22;
        v15 = v18;
      }
      sub_B99460(v7, v17, v15, v16);
      v5 = (__int64 *)v5[1];
    }
    while ( v3 != (__int64 **)v5 );
  }
LABEL_15:
  memset((void *)a1, 0, 0x60u);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
