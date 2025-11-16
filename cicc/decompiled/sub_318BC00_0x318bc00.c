// Function: sub_318BC00
// Address: 0x318bc00
//
_QWORD *__fastcall sub_318BC00(
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
  __int64 v11; // rax
  unsigned __int8 *v12; // r9
  _BYTE *v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // r13
  _BYTE *v16; // r15
  __int64 (__fastcall *v17)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v18; // rax
  _BYTE *v19; // r12
  __int64 v20; // rsi
  __int64 v21; // rdi
  _QWORD *v23; // rax
  __int64 v24; // r9
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // rax
  unsigned __int8 *v30; // [rsp+0h] [rbp-80h]
  unsigned __int8 *v31; // [rsp+0h] [rbp-80h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  char v34[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v35; // [rsp+40h] [rbp-40h]

  v11 = sub_318B710(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v12 = *(unsigned __int8 **)(a3 + 16);
  v13 = *(_BYTE **)(a1 + 16);
  v14 = *(_QWORD *)(v11 + 80);
  v15 = v11;
  v16 = *(_BYTE **)(a2 + 16);
  v17 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v14 + 104LL);
  if ( v17 != sub_948040 )
  {
    v31 = v12;
    v29 = v17(v14, v13, v16, v12);
    v12 = v31;
    v19 = (_BYTE *)v29;
LABEL_6:
    if ( v19 )
      goto LABEL_7;
    goto LABEL_9;
  }
  if ( *v13 <= 0x15u && *v16 <= 0x15u && *v12 <= 0x15u )
  {
    v30 = v12;
    v18 = sub_AD5A90((__int64)v13, *(_BYTE **)(a2 + 16), v12, 0);
    v12 = v30;
    v19 = (_BYTE *)v18;
    goto LABEL_6;
  }
LABEL_9:
  v32 = (__int64)v12;
  v35 = 257;
  v23 = sub_BD2C40(72, 3u);
  v24 = v32;
  v19 = v23;
  if ( v23 )
    sub_B4DFA0((__int64)v23, (__int64)v13, (__int64)v16, v32, (__int64)v34, v32, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, __int64, _QWORD, _QWORD, __int64))(**(_QWORD **)(v15 + 88) + 16LL))(
    *(_QWORD *)(v15 + 88),
    v19,
    a5,
    *(_QWORD *)(v15 + 56),
    *(_QWORD *)(v15 + 64),
    v24);
  v25 = *(_QWORD *)v15;
  v26 = *(_QWORD *)v15 + 16LL * *(unsigned int *)(v15 + 8);
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
    v21 = a4;
    if ( *v19 != 91 )
      return (_QWORD *)sub_31892C0(v21, v20);
    return sub_3189A90(v21, v20);
  }
LABEL_7:
  v20 = (__int64)v19;
  v21 = a4;
  if ( *v19 != 91 )
    return (_QWORD *)sub_31892C0(v21, v20);
  return sub_3189A90(v21, v20);
}
