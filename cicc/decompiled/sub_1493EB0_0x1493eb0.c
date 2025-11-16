// Function: sub_1493EB0
// Address: 0x1493eb0
//
__int64 __fastcall sub_1493EB0(
        __int64 a1,
        _QWORD *a2,
        _QWORD *a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        __m128i a7,
        __m128i a8,
        char a9)
{
  __int16 v12; // r15
  __int64 v13; // rsi
  unsigned int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v26; // [rsp+20h] [rbp-D0h]
  __int64 v27; // [rsp+20h] [rbp-D0h]
  unsigned int v28; // [rsp+28h] [rbp-C8h]
  unsigned __int8 v29; // [rsp+28h] [rbp-C8h]
  unsigned int v30; // [rsp+28h] [rbp-C8h]
  unsigned __int8 v31; // [rsp+28h] [rbp-C8h]
  unsigned int v32; // [rsp+3Ch] [rbp-B4h] BYREF
  __int64 v33; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v35; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+58h] [rbp-98h] BYREF
  __int64 v37; // [rsp+60h] [rbp-90h] BYREF
  __int64 v38; // [rsp+68h] [rbp-88h]
  __int64 v39; // [rsp+70h] [rbp-80h] BYREF
  _BYTE v40[8]; // [rsp+78h] [rbp-78h] BYREF
  __int64 v41; // [rsp+80h] [rbp-70h]
  unsigned __int64 v42; // [rsp+88h] [rbp-68h]

  v12 = *(_WORD *)(a4 + 18);
  if ( a5 )
  {
    v21 = sub_15FF0F0(v12 & 0x7FFF);
    v13 = *(_QWORD *)(a4 - 48);
    v32 = v21;
    v14 = v21;
    if ( *(_BYTE *)(v13 + 16) != 54 )
      goto LABEL_3;
  }
  else
  {
    v13 = *(_QWORD *)(a4 - 48);
    v14 = v12 & 0x7FFF;
    v32 = v14;
    if ( *(_BYTE *)(v13 + 16) != 54 )
      goto LABEL_3;
  }
  if ( *(_BYTE *)(*(_QWORD *)(a4 - 24) + 16LL) <= 0x10u )
  {
    sub_1487E90((__int64)&v37, (__int64)a2, v13);
    if ( !sub_14562D0(v37) || !sub_14562D0(v38) )
      goto LABEL_27;
    if ( v42 != v41 )
      _libc_free(v42);
    v13 = *(_QWORD *)(a4 - 48);
  }
LABEL_3:
  v15 = sub_146F1B0((__int64)a2, v13);
  v16 = *(_QWORD *)(a4 - 24);
  v33 = v15;
  v34 = sub_146F1B0((__int64)a2, v16);
  v33 = sub_1472270((__int64)a2, v33, a3);
  v34 = sub_1472270((__int64)a2, v34, a3);
  if ( sub_146CEE0((__int64)a2, v33, (__int64)a3) && !sub_146CEE0((__int64)a2, v34, (__int64)a3) )
  {
    v22 = v33;
    v33 = v34;
    v34 = v22;
    v32 = sub_15FF5D0(v32);
  }
  sub_147DF40((__int64)a2, &v32, &v33, &v34, 0, v17);
  if ( !*(_WORD *)(v34 + 24) && *(_WORD *)(v33 + 24) == 7 && a3 == *(_QWORD **)(v33 + 48) )
  {
    v26 = v33;
    sub_158B890(&v37, v32, *(_QWORD *)(v34 + 32) + 24LL);
    v27 = sub_1488A30(v26, (__int64)&v37, a2, a7, a8);
    if ( !sub_14562D0(v27) )
    {
      sub_14573F0(a1, v27);
      sub_135E100(&v39);
      sub_135E100(&v37);
      return a1;
    }
    sub_135E100(&v39);
    sub_135E100(&v37);
  }
  switch ( v32 )
  {
    case ' ':
      v23 = sub_14806B0((__int64)a2, v33, v34, 0, 0);
      sub_145CFC0((__int64)&v37, (__int64)a2, v23);
      if ( !sub_14562D0(v37) )
        goto LABEL_27;
      goto LABEL_9;
    case '!':
      v18 = sub_14806B0((__int64)a2, v33, v34, 0, 0);
      sub_1472640((__int64)&v37, (__int64)a2, v18, (_QWORD **)a3, a6, a9);
      if ( !sub_14562D0(v37) )
        goto LABEL_27;
      goto LABEL_9;
    case '"':
    case '&':
      sub_14937C0((__int64)&v37, a2, v33, v34, (__int64)a3, v32 == 38, a7, a8, a6, a9);
      break;
    case '#':
    case '\'':
      v35 = sub_1480810((__int64)a2, v33);
      v36 = sub_1480810((__int64)a2, v34);
      if ( !byte_4F9A540 )
        goto LABEL_12;
      v30 = v32;
      if ( !sub_146CEE0((__int64)a2, v36, (__int64)a3) )
        goto LABEL_12;
      v31 = v30 == 39;
      if ( !(unsigned __int8)sub_14809B0(&v35, &v36, (__int64)a3, v31, (__int64)a2) )
        goto LABEL_12;
      sub_14937C0((__int64)&v37, a2, v35, v36, (__int64)a3, v31, a7, a8, a9, 0);
      break;
    case '$':
    case '(':
      sub_1493280((__int64)&v37, a2, v33, v34, (__int64)a3, v32 == 40, a7, a8, a6, a9);
      break;
    case '%':
    case ')':
      v28 = v32;
      if ( !byte_4F9A540 )
        goto LABEL_12;
      if ( !sub_146CEE0((__int64)a2, v34, (__int64)a3) )
        goto LABEL_12;
      v29 = v28 == 41;
      if ( !(unsigned __int8)sub_14809B0(&v33, &v34, (__int64)a3, v29, (__int64)a2) )
        goto LABEL_12;
      sub_14937C0((__int64)&v37, a2, v33, v34, (__int64)a3, v29, a7, a8, a9, 0);
      break;
    default:
      goto LABEL_12;
  }
  if ( !sub_14562D0(v37) )
    goto LABEL_27;
LABEL_9:
  if ( !sub_14562D0(v38) )
  {
LABEL_27:
    *(_QWORD *)a1 = v37;
    *(_QWORD *)(a1 + 8) = v38;
    *(_BYTE *)(a1 + 16) = v39;
    sub_16CCEE0(a1 + 24, a1 + 64, 4, v40);
    if ( v42 != v41 )
      _libc_free(v42);
    return a1;
  }
  if ( v42 != v41 )
    _libc_free(v42);
LABEL_12:
  v19 = sub_146AAD0((__int64)a2, (__int64)a3, a4, a5);
  if ( sub_14562D0(v19) )
    sub_145D330(a1, a2, *(_QWORD *)(a4 - 48), *(__int64 **)(a4 - 24), (__int64)a3, v14);
  else
    sub_14573F0(a1, v19);
  return a1;
}
