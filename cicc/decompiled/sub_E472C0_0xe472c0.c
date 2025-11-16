// Function: sub_E472C0
// Address: 0xe472c0
//
unsigned __int64 *__fastcall sub_E472C0(
        unsigned __int64 *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        __m128i a6)
{
  _QWORD *v8; // rsi
  _BYTE *v9; // rax
  bool v11; // zf
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 *v15; // rsi
  char v16; // al
  __int64 v17; // r8
  __int64 *v18; // rax
  __int64 *v19; // rcx
  __int64 *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rax
  _QWORD **v25; // r13
  __int64 *v26; // [rsp+8h] [rbp-128h]
  __int64 v27; // [rsp+20h] [rbp-110h]
  __int64 *v28; // [rsp+20h] [rbp-110h]
  int v29; // [rsp+28h] [rbp-108h]
  __int64 v30; // [rsp+28h] [rbp-108h]
  __int64 *v31; // [rsp+28h] [rbp-108h]
  __int64 v32; // [rsp+28h] [rbp-108h]
  __int64 v33; // [rsp+30h] [rbp-100h] BYREF
  __int64 v34; // [rsp+38h] [rbp-F8h] BYREF
  __int64 v35; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v36; // [rsp+48h] [rbp-E8h]
  __int64 v37; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v38; // [rsp+58h] [rbp-D8h] BYREF
  __int64 v39; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v40; // [rsp+68h] [rbp-C8h] BYREF
  _QWORD **v41; // [rsp+70h] [rbp-C0h] BYREF
  char v42; // [rsp+78h] [rbp-B8h]
  __int64 v43; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v44; // [rsp+88h] [rbp-A8h]
  void (__fastcall *v45)(__int64 *, __int64 *, __int64); // [rsp+90h] [rbp-A0h]
  __int64 v46; // [rsp+98h] [rbp-98h]
  _QWORD v47[18]; // [rsp+A0h] [rbp-90h] BYREF

  v8 = (_QWORD *)*a2;
  v9 = (_BYTE *)v8[1];
  if ( v9 == (_BYTE *)v8[2] )
    goto LABEL_5;
  if ( *v9 != 0xDE )
  {
    if ( *v9 == 66 && v9[1] == 67 && v9[2] == 0xC0 && v9[3] == 0xDE )
      goto LABEL_13;
LABEL_5:
    v29 = a4;
    sub_C7E010(&v43, v8);
    sub_1060120(
      (_DWORD)a1,
      a3,
      v29,
      0,
      (unsigned int)sub_E45A70,
      (unsigned int)&v41,
      v43,
      (__int64)v44,
      (__int64)v45,
      v46);
    return a1;
  }
  if ( v9[1] != 0xC0 || v9[2] != 23 || v9[3] != 11 )
    goto LABEL_5;
LABEL_13:
  memset(v47, 0, 0x58u);
  sub_A01490((__int64)&v41, a2, a4, a5, 0, (__int64)&v43, a6);
  if ( LOBYTE(v47[10]) && (LOBYTE(v47[10]) = 0, v47[8]) )
  {
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v47[8])(&v47[6], &v47[6], 3);
    if ( !LOBYTE(v47[5]) )
      goto LABEL_15;
  }
  else if ( !LOBYTE(v47[5]) )
  {
    goto LABEL_15;
  }
  LOBYTE(v47[5]) = 0;
  if ( v47[3] )
  {
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v47[3])(&v47[1], &v47[1], 3);
    if ( !LOBYTE(v47[0]) )
      goto LABEL_16;
    goto LABEL_44;
  }
LABEL_15:
  if ( !LOBYTE(v47[0]) )
    goto LABEL_16;
LABEL_44:
  LOBYTE(v47[0]) = 0;
  if ( v45 )
    v45(&v43, &v43, 3);
LABEL_16:
  v11 = (v42 & 1) == 0;
  v12 = v42 & 1;
  v42 &= ~2u;
  v13 = (unsigned __int64)v41;
  if ( v11 )
  {
LABEL_39:
    *a1 = v13;
    return a1;
  }
  v41 = 0;
  v14 = v13 & 0xFFFFFFFFFFFFFFFELL;
  if ( !v14 )
  {
    v13 = 0;
    goto LABEL_39;
  }
  v43 = a3;
  v15 = (__int64 *)&unk_4F84052;
  v44 = a2;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v30 = v14;
  v16 = (*(__int64 (__fastcall **)(unsigned __int64, void *, __int64))(*(_QWORD *)v14 + 48LL))(v14, &unk_4F84052, v12);
  v17 = v30;
  if ( v16 )
  {
    v18 = *(__int64 **)(v30 + 16);
    v19 = *(__int64 **)(v30 + 8);
    v36 = 1;
    v26 = v18;
    if ( v19 == v18 )
    {
      v21 = 1;
    }
    else
    {
      do
      {
        v27 = v17;
        v38 = *v19;
        *v19 = 0;
        v31 = v19;
        sub_E47240(&v39, &v38, (__int64)&v43);
        v15 = &v37;
        v37 = v36 | 1;
        sub_9CDB40((unsigned __int64 *)&v40, (unsigned __int64 *)&v37, (unsigned __int64 *)&v39);
        v20 = v31;
        v17 = v27;
        v36 = v40 | 1;
        if ( (v37 & 1) != 0 || (v37 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v37, (__int64)&v37);
        if ( (v39 & 1) != 0 || (v39 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v39, (__int64)&v37);
        if ( v38 )
        {
          v28 = v31;
          v32 = v17;
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v38 + 8LL))(v38);
          v20 = v28;
          v17 = v32;
        }
        v19 = v20 + 1;
      }
      while ( v26 != v19 );
      v21 = v36 | 1;
    }
    v39 = v21;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
  }
  else
  {
    v15 = &v40;
    v40 = v30;
    sub_E47240(&v39, &v40, (__int64)&v43);
    if ( v40 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v40 + 8LL))(v40);
  }
  if ( (v39 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  if ( (v35 & 1) != 0 || (v35 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v35, (__int64)v15);
  if ( (v34 & 1) != 0 || (v34 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v34, (__int64)v15);
  v24 = v33;
  *a1 = 0;
  if ( (v24 & 1) != 0 || (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v33, (__int64)v15);
  if ( (v42 & 2) != 0 )
    sub_904700(&v41);
  v25 = v41;
  if ( (v42 & 1) != 0 )
  {
    if ( v41 )
      ((void (__fastcall *)(_QWORD **))(*v41)[1])(v41);
  }
  else if ( v41 )
  {
    sub_BA9C10(v41, (__int64)v15, v22, v23);
    j_j___libc_free_0(v25, 880);
  }
  return a1;
}
