// Function: sub_1DC1A70
// Address: 0x1dc1a70
//
void __fastcall sub_1DC1A70(_QWORD *a1, unsigned __int64 a2, char a3)
{
  unsigned __int64 v4; // rdx
  __int64 i; // rdi
  __int64 v8; // rcx
  __int64 v9; // rsi
  unsigned int v10; // r8d
  __int64 *v11; // rax
  __int64 v12; // r10
  __int64 v13; // r14
  unsigned __int64 v14; // rax
  unsigned __int8 *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // eax
  int v19; // r11d
  _QWORD v20[6]; // [rsp+0h] [rbp-C0h] BYREF
  _BYTE *v21; // [rsp+30h] [rbp-90h]
  _BYTE *v22; // [rsp+38h] [rbp-88h]
  __int64 v23; // [rsp+40h] [rbp-80h]
  int v24; // [rsp+48h] [rbp-78h]
  _BYTE v25[112]; // [rsp+50h] [rbp-70h] BYREF

  v4 = a2;
  for ( i = a1[34]; (*(_BYTE *)(v4 + 46) & 4) != 0; v4 = *(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v8 = *(unsigned int *)(i + 384);
  v9 = *(_QWORD *)(i + 368);
  if ( (_DWORD)v8 )
  {
    v10 = (v8 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( v4 == *v11 )
      goto LABEL_5;
    v18 = 1;
    while ( v12 != -8 )
    {
      v19 = v18 + 1;
      v10 = (v8 - 1) & (v18 + v10);
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v4 == *v11 )
        goto LABEL_5;
      v18 = v19;
    }
  }
  v11 = (__int64 *)(v9 + 16 * v8);
LABEL_5:
  v13 = v11[1];
  sub_1F10740(i, a2);
  v14 = sub_1DC1550(a1[34], a2, 0);
  v15 = *(unsigned __int8 **)(a2 + 32);
  v25[64] = a3;
  v16 = a1[31];
  v17 = a1[30];
  v20[4] = v14;
  v20[0] = a1;
  v20[2] = v16;
  LODWORD(v16) = *(_DWORD *)(a2 + 40);
  v20[1] = v17;
  v20[3] = v13;
  v20[5] = 0;
  v21 = v25;
  v22 = v25;
  v23 = 8;
  v24 = 0;
  sub_1DBE170((__int64)v20, v15, v16);
  if ( v22 != v21 )
    _libc_free((unsigned __int64)v22);
}
