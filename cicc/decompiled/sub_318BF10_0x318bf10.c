// Function: sub_318BF10
// Address: 0x318bf10
//
_QWORD *__fastcall sub_318BF10(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v12; // rax
  _BYTE *v13; // r13
  _BYTE *v14; // r10
  __int64 v15; // rdi
  __int64 v16; // rbx
  __int64 (__fastcall *v17)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v18; // rax
  _BYTE *v19; // r14
  __int64 v20; // rsi
  __int64 v21; // rdi
  _QWORD *v23; // rax
  __int64 v24; // r13
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+8h] [rbp-78h]
  _BYTE *v32; // [rsp+8h] [rbp-78h]
  _BYTE v35[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v36; // [rsp+40h] [rbp-40h]

  v12 = sub_318B710(a1, a2, (__int64)a3, a4, a5, a6, a7, a8, a9);
  v13 = *(_BYTE **)(a2 + 16);
  v14 = *(_BYTE **)(a1 + 16);
  v15 = *(_QWORD *)(v12 + 80);
  v16 = v12;
  v17 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v15 + 112LL);
  if ( v17 != sub_9B6630 )
  {
    v32 = *(_BYTE **)(a1 + 16);
    v29 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _DWORD *, __int64))v17)(v15, v14, v13, a3, a4);
    v14 = v32;
    v19 = (_BYTE *)v29;
LABEL_5:
    if ( v19 )
      goto LABEL_6;
    goto LABEL_8;
  }
  if ( *v14 <= 0x15u && *v13 <= 0x15u )
  {
    v30 = *(_QWORD *)(a1 + 16);
    v18 = sub_AD5CE0(v30, (__int64)v13, a3, a4, 0);
    v14 = (_BYTE *)v30;
    v19 = (_BYTE *)v18;
    goto LABEL_5;
  }
LABEL_8:
  v31 = (__int64)v14;
  v36 = 257;
  v23 = sub_BD2C40(112, unk_3F1FE60);
  v19 = v23;
  if ( v23 )
    sub_B4E9E0((__int64)v23, v31, (__int64)v13, a3, a4, (__int64)v35, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, __int64, _QWORD, _QWORD))(**(_QWORD **)(v16 + 88) + 16LL))(
    *(_QWORD *)(v16 + 88),
    v19,
    a6,
    *(_QWORD *)(v16 + 56),
    *(_QWORD *)(v16 + 64));
  v24 = 16LL * *(unsigned int *)(v16 + 8);
  v25 = *(_QWORD *)v16;
  v26 = v25 + v24;
  if ( v25 != v26 )
  {
    do
    {
      v27 = *(_QWORD *)(v25 + 8);
      v28 = *(_DWORD *)v25;
      v25 += 16;
      sub_B99FD0((__int64)v19, v28, v27);
    }
    while ( v26 != v25 );
    v20 = (__int64)v19;
    v21 = a5;
    if ( *v19 != 92 )
      return (_QWORD *)sub_31892C0(v21, v20);
    return sub_3189B10(v21, v20);
  }
LABEL_6:
  v20 = (__int64)v19;
  v21 = a5;
  if ( *v19 != 92 )
    return (_QWORD *)sub_31892C0(v21, v20);
  return sub_3189B10(v21, v20);
}
