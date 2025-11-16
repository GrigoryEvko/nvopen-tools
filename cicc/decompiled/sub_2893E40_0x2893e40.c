// Function: sub_2893E40
// Address: 0x2893e40
//
__int64 __fastcall sub_2893E40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  _BYTE *v8; // r14
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 *v13; // rdi
  __int64 v14; // r13
  _QWORD **v15; // rbx
  _BYTE *v16; // r14
  _QWORD *v17; // r15
  __int64 v19; // [rsp+18h] [rbp-88h] BYREF
  _BYTE *v20; // [rsp+20h] [rbp-80h] BYREF
  __int64 v21; // [rsp+28h] [rbp-78h]
  _BYTE v22[112]; // [rsp+30h] [rbp-70h] BYREF

  v5 = sub_B6AC80(*(_QWORD *)(a3 + 40), 153);
  if ( !v5 || (v6 = v5, !*(_QWORD *)(v5 + 16)) || (v7 = *(_QWORD *)(v5 + 16), v20 = v22, v21 = 0x800000000LL, !v7) )
  {
LABEL_21:
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
  do
  {
    while ( 1 )
    {
      v8 = *(_BYTE **)(v7 + 24);
      if ( *v8 == 85 && a3 == sub_B43CB0(*(_QWORD *)(v7 + 24)) )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        goto LABEL_11;
    }
    v11 = (unsigned int)v21;
    v12 = (unsigned int)v21 + 1LL;
    if ( v12 > HIDWORD(v21) )
    {
      sub_C8D5F0((__int64)&v20, v22, v12, 8u, v9, v10);
      v11 = (unsigned int)v21;
    }
    *(_QWORD *)&v20[8 * v11] = v8;
    LODWORD(v21) = v21 + 1;
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v7 );
LABEL_11:
  if ( !(_DWORD)v21 )
  {
    if ( v20 != v22 )
      _libc_free((unsigned __int64)v20);
    goto LABEL_21;
  }
  v13 = *(__int64 **)(a3 + 40);
  v19 = **(_QWORD **)(*(_QWORD *)(a3 + 24) + 16LL);
  v14 = sub_B6E160(v13, 0x92u, (__int64)&v19, 1);
  *(_WORD *)(v14 + 2) = *(_WORD *)(v14 + 2) & 0xC00F | *(_WORD *)(v6 + 2) & 0x3FF0;
  v15 = (_QWORD **)v20;
  v16 = &v20[8 * (unsigned int)v21];
  if ( v20 != v16 )
  {
    do
    {
      v17 = *v15++;
      sub_29DD780(v14, v17, 0);
      sub_B43D60(v17);
    }
    while ( v16 != (_BYTE *)v15 );
    v16 = v20;
  }
  if ( v16 != v22 )
    _libc_free((unsigned __int64)v16);
  memset((void *)a1, 0, 0x60u);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
