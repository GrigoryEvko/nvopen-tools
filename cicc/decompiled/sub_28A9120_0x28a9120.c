// Function: sub_28A9120
// Address: 0x28a9120
//
__int64 __fastcall sub_28A9120(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rax
  _QWORD *v6; // r14
  _QWORD *v7; // rbx
  _QWORD *i; // r13
  __int64 *v9; // rdi
  __int64 v10; // r15
  _QWORD **v11; // rbx
  _BYTE *v12; // r13
  _QWORD *v13; // r14
  __int64 v15; // r11
  bool v16; // al
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r11
  _QWORD *v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v24; // [rsp+10h] [rbp-A0h]
  __int64 v25; // [rsp+18h] [rbp-98h]
  __int64 v26; // [rsp+28h] [rbp-88h] BYREF
  _BYTE *v27; // [rsp+30h] [rbp-80h] BYREF
  __int64 v28; // [rsp+38h] [rbp-78h]
  _BYTE v29[112]; // [rsp+40h] [rbp-70h] BYREF

  v5 = sub_B6AC80(a3[5], 153);
  v24 = v5;
  if ( !v5 || !*(_QWORD *)(v5 + 16) )
  {
LABEL_36:
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v6 = a3 + 9;
  v7 = (_QWORD *)a3[10];
  v27 = v29;
  v28 = 0x800000000LL;
  if ( v6 == v7 )
  {
    i = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    while ( 1 )
    {
      i = (_QWORD *)v7[4];
      if ( i != v7 + 3 )
        break;
      v7 = (_QWORD *)v7[1];
      if ( v6 == v7 )
        goto LABEL_9;
      if ( !v7 )
        BUG();
    }
  }
  while ( v6 != v7 )
  {
    v15 = (__int64)(i - 3);
    if ( !i )
      v15 = 0;
    v25 = v15;
    v16 = sub_D222C0(v15);
    v19 = v25;
    if ( v16 )
    {
      v21 = (unsigned int)v28;
      v22 = (unsigned int)v28 + 1LL;
      if ( v22 > HIDWORD(v28) )
      {
        sub_C8D5F0((__int64)&v27, v29, v22, 8u, v17, v18);
        v21 = (unsigned int)v28;
        v19 = v25;
      }
      *(_QWORD *)&v27[8 * v21] = v19;
      LODWORD(v28) = v28 + 1;
    }
    for ( i = (_QWORD *)i[1]; ; i = (_QWORD *)v7[4] )
    {
      v20 = v7 - 3;
      if ( !v7 )
        v20 = 0;
      if ( i != v20 + 6 )
        break;
      v7 = (_QWORD *)v7[1];
      if ( v6 == v7 )
        goto LABEL_9;
      if ( !v7 )
        BUG();
    }
  }
LABEL_9:
  if ( !(_DWORD)v28 )
  {
    if ( v27 != v29 )
      _libc_free((unsigned __int64)v27);
    goto LABEL_36;
  }
  v9 = (__int64 *)a3[5];
  v26 = **(_QWORD **)(a3[3] + 16LL);
  v10 = sub_B6E160(v9, 0x92u, (__int64)&v26, 1);
  *(_WORD *)(v10 + 2) = *(_WORD *)(v10 + 2) & 0xC00F | *(_WORD *)(v24 + 2) & 0x3FF0;
  v11 = (_QWORD **)v27;
  v12 = &v27[8 * (unsigned int)v28];
  if ( v27 != v12 )
  {
    do
    {
      v13 = *v11++;
      sub_29DD780(v10, v13, 1);
      sub_B43D60(v13);
    }
    while ( v12 != (_BYTE *)v11 );
    v12 = v27;
  }
  if ( v12 != v29 )
    _libc_free((unsigned __int64)v12);
  memset((void *)a1, 0, 0x60u);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
