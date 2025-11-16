// Function: sub_260FA60
// Address: 0x260fa60
//
__int64 __fastcall sub_260FA60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // r8
  _QWORD *v11; // r14
  __int64 v12; // rax
  char v13; // al
  __int64 v15; // [rsp+0h] [rbp-40h]
  char v16; // [rsp+Fh] [rbp-31h]

  v5 = a3 + 24;
  v6 = sub_BC0510(a4, &unk_4F82418, a3);
  v7 = *(_QWORD *)(a3 + 32);
  v8 = a1 + 32;
  v9 = a1 + 80;
  v15 = *(_QWORD *)(v6 + 8);
  if ( v5 == v7 )
    goto LABEL_14;
  v16 = 0;
  do
  {
    while ( 1 )
    {
      v10 = v7 - 56;
      if ( !v7 )
        v10 = 0;
      v11 = (_QWORD *)v10;
      if ( sub_B2FC80(v10) && !(unsigned __int8)sub_B2D610((__int64)v11, 48) )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v5 == v7 )
        goto LABEL_11;
    }
    if ( !(unsigned __int8)sub_B2D610((__int64)v11, 23) )
    {
      v12 = sub_BC1CD0(v15, &unk_4F6D3F8, (__int64)v11);
      v16 |= sub_11C5620(v11, (__int64 *)(v12 + 8));
    }
    v13 = sub_F58C20((__int64)v11);
    v7 = *(_QWORD *)(v7 + 8);
    v16 |= v13;
  }
  while ( v5 != v7 );
LABEL_11:
  v8 = a1 + 32;
  v9 = a1 + 80;
  if ( !v16 )
  {
LABEL_14:
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v9;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    memset((void *)a1, 0, 0x60u);
    *(_QWORD *)(a1 + 8) = v8;
    *(_DWORD *)(a1 + 16) = 2;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 56) = v9;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
  }
  return a1;
}
