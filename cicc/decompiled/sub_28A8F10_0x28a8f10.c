// Function: sub_28A8F10
// Address: 0x28a8f10
//
__int64 __fastcall sub_28A8F10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  _BYTE *v6; // r15
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  _QWORD **v11; // r14
  _QWORD **v12; // rbx
  _QWORD *v13; // r15
  __int64 *v14; // rax
  __int64 v15; // rax
  _BYTE *v17; // [rsp+10h] [rbp-80h] BYREF
  __int64 v18; // [rsp+18h] [rbp-78h]
  _BYTE v19[112]; // [rsp+20h] [rbp-70h] BYREF

  v4 = sub_B6AC80(*(_QWORD *)(a3 + 40), 169);
  if ( !v4 || !*(_QWORD *)(v4 + 16) || (v5 = *(_QWORD *)(v4 + 16), v17 = v19, v18 = 0x800000000LL, !v5) )
  {
LABEL_20:
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
      v6 = *(_BYTE **)(v5 + 24);
      if ( *v6 == 85 && a3 == sub_B43CB0(*(_QWORD *)(v5 + 24)) )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        goto LABEL_11;
    }
    v9 = (unsigned int)v18;
    v10 = (unsigned int)v18 + 1LL;
    if ( v10 > HIDWORD(v18) )
    {
      sub_C8D5F0((__int64)&v17, v19, v10, 8u, v7, v8);
      v9 = (unsigned int)v18;
    }
    *(_QWORD *)&v17[8 * v9] = v6;
    LODWORD(v18) = v18 + 1;
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v5 );
LABEL_11:
  if ( !(_DWORD)v18 )
  {
    if ( v17 != v19 )
      _libc_free((unsigned __int64)v17);
    goto LABEL_20;
  }
  v11 = (_QWORD **)&v17[8 * (unsigned int)v18];
  v12 = (_QWORD **)v17;
  do
  {
    v13 = *v12++;
    v14 = (__int64 *)sub_BD5C60((__int64)v13);
    v15 = sub_ACD6D0(v14);
    sub_BD84D0((__int64)v13, v15);
    sub_B43D60(v13);
  }
  while ( v11 != v12 );
  if ( v17 != v19 )
    _libc_free((unsigned __int64)v17);
  memset((void *)a1, 0, 0x60u);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
