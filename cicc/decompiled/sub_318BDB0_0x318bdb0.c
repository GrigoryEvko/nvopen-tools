// Function: sub_318BDB0
// Address: 0x318bdb0
//
_QWORD *__fastcall sub_318BDB0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // rax
  _BYTE *v11; // r15
  _BYTE *v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // r13
  __int64 (__fastcall *v15)(__int64, _BYTE *, _BYTE *); // rax
  _BYTE *v16; // r12
  __int64 v17; // rsi
  __int64 v18; // rdi
  _QWORD *v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rdx
  unsigned int v24; // esi
  char v26[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v27; // [rsp+30h] [rbp-40h]

  v10 = sub_318B710(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v11 = *(_BYTE **)(a2 + 16);
  v12 = *(_BYTE **)(a1 + 16);
  v13 = *(_QWORD *)(v10 + 80);
  v14 = v10;
  v15 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *))(*(_QWORD *)v13 + 96LL);
  if ( v15 != sub_948070 )
  {
    v16 = (_BYTE *)v15(v13, v12, *(_BYTE **)(a2 + 16));
LABEL_5:
    if ( v16 )
      goto LABEL_6;
    goto LABEL_8;
  }
  if ( *v12 <= 0x15u && *v11 <= 0x15u )
  {
    v16 = (_BYTE *)sub_AD5840((__int64)v12, *(unsigned __int8 **)(a2 + 16), 0);
    goto LABEL_5;
  }
LABEL_8:
  v27 = 257;
  v20 = sub_BD2C40(72, 2u);
  v16 = v20;
  if ( v20 )
    sub_B4DE80((__int64)v20, (__int64)v12, (__int64)v11, (__int64)v26, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, __int64, _QWORD, _QWORD))(**(_QWORD **)(v14 + 88) + 16LL))(
    *(_QWORD *)(v14 + 88),
    v16,
    a4,
    *(_QWORD *)(v14 + 56),
    *(_QWORD *)(v14 + 64));
  v21 = *(_QWORD *)v14;
  v22 = *(_QWORD *)v14 + 16LL * *(unsigned int *)(v14 + 8);
  if ( v21 != v22 )
  {
    do
    {
      v23 = *(_QWORD *)(v21 + 8);
      v24 = *(_DWORD *)v21;
      v21 += 16;
      sub_B99FD0((__int64)v16, v24, v23);
    }
    while ( v22 != v21 );
    v17 = (__int64)v16;
    v18 = a3;
    if ( *v16 != 90 )
      return (_QWORD *)sub_31892C0(v18, v17);
    return sub_3189A10(v18, v17);
  }
LABEL_6:
  v17 = (__int64)v16;
  v18 = a3;
  if ( *v16 != 90 )
    return (_QWORD *)sub_31892C0(v18, v17);
  return sub_3189A10(v18, v17);
}
