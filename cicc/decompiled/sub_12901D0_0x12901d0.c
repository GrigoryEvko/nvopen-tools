// Function: sub_12901D0
// Address: 0x12901d0
//
_QWORD *__fastcall sub_12901D0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 *v11; // rsi
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 (__fastcall *v14)(__int64 **, __int64, char *, __int64, _BOOL8); // r10
  __int64 v15; // rsi
  _BOOL8 v16; // r8
  __int64 v17; // rcx
  unsigned __int8 v18; // al
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 *v24; // rdi
  __int64 v25; // rcx
  _QWORD *result; // rax
  __int64 *v27; // rsi
  __int64 i; // rax
  char v29; // dl
  __int64 *v30; // [rsp+8h] [rbp-138h]
  _QWORD *v32; // [rsp+18h] [rbp-128h]
  char *v33; // [rsp+20h] [rbp-120h]
  char v34; // [rsp+28h] [rbp-118h]
  int v35; // [rsp+2Ch] [rbp-114h]
  unsigned __int64 v36; // [rsp+38h] [rbp-108h]
  _QWORD *v37; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v38[5]; // [rsp+50h] [rbp-F0h] BYREF
  int v39; // [rsp+78h] [rbp-C8h]
  char v40; // [rsp+7Ch] [rbp-C4h]
  __int64 v41; // [rsp+80h] [rbp-C0h]
  _QWORD *v42; // [rsp+90h] [rbp-B0h] BYREF
  int v43; // [rsp+98h] [rbp-A8h]
  char v44; // [rsp+9Ch] [rbp-A4h]
  __int64 v45; // [rsp+A0h] [rbp-A0h]
  __int64 v46; // [rsp+B0h] [rbp-90h] BYREF
  _BYTE *v47; // [rsp+B8h] [rbp-88h]
  __int64 v48; // [rsp+C0h] [rbp-80h]
  __int64 v49; // [rsp+C8h] [rbp-78h]
  __int64 v50; // [rsp+D0h] [rbp-70h]
  __int64 v51; // [rsp+D8h] [rbp-68h]
  __int64 v52; // [rsp+E0h] [rbp-60h]
  _BYTE *v53; // [rsp+E8h] [rbp-58h]
  __int64 v54; // [rsp+F0h] [rbp-50h]
  __int64 v55; // [rsp+F8h] [rbp-48h]
  __int64 v56; // [rsp+100h] [rbp-40h]
  __int64 v57; // [rsp+108h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  v30 = *(__int64 **)(v7 + 16);
  v33 = sub_128D0F0(a1, (__int64)v30, a3, a4, a5);
  v36 = *(_QWORD *)a2;
  v8 = sub_127B450(a2);
  sub_1286D80((__int64)&v46, *a1, v7, v9, v10);
  v35 = v46;
  v54 = v48;
  v11 = *a1;
  v53 = v47;
  v55 = v49;
  v56 = v50;
  v57 = v51;
  v34 = v51;
  v52 = v46;
  sub_1287CD0((__int64)&v42, v11, (_DWORD *)(v7 + 36), (unsigned int)v51, v50, v51, v46, v47, v48, v49, v50, v51);
  v32 = v42;
  v12 = sub_127B3A0(v36);
  v13 = sub_128B370((__int64 *)a1, v32, v12, v8, (_DWORD *)(a2 + 36));
  v14 = (__int64 (__fastcall *)(__int64 **, __int64, char *, __int64, _BOOL8))a3;
  v15 = v13;
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 56) - 84) <= 1u )
  {
    if ( (a3 & 1) != 0 )
      v14 = *(__int64 (__fastcall **)(__int64 **, __int64, char *, __int64, _BOOL8))((char *)*a1 + a3 - 1);
    for ( i = *(_QWORD *)v7; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    do
    {
      i = *(_QWORD *)(i + 160);
      v29 = *(_BYTE *)(i + 140);
    }
    while ( v29 == 12 );
    v16 = v29 == 1;
    v17 = *v30;
  }
  else
  {
    if ( (a3 & 1) != 0 )
      v14 = *(__int64 (__fastcall **)(__int64 **, __int64, char *, __int64, _BOOL8))((char *)*a1 + a3 - 1);
    v16 = 0;
    v17 = v8;
  }
  v37 = (_QWORD *)v14(a1, v15, v33, v17, v16);
  v18 = sub_127B3A0(v8);
  v19 = (_QWORD *)sub_128B370((__int64 *)a1, v37, v18, v36, (_DWORD *)(a2 + 36));
  v24 = *a1;
  v37 = v19;
  if ( v35 == 1 )
  {
    if ( (v34 & 1) == 0 )
    {
      BYTE4(v53) &= ~1u;
      LODWORD(v53) = 0;
      LODWORD(v54) = 0;
      LODWORD(v46) = 1;
      v52 = (__int64)v19;
      sub_1282050(
        v24,
        (_DWORD *)(v7 + 36),
        (__int64 *)&v37,
        v21,
        v22,
        v23,
        (__int64)v19,
        0,
        0,
        v46,
        v47,
        v48,
        v49,
        v50,
        v51);
      return v37;
    }
    v44 &= ~1u;
    v43 = 0;
    LODWORD(v45) = 0;
    LODWORD(v46) = 1;
    v42 = v19;
    sub_1282050(v24, (_DWORD *)(v7 + 36), 0, v21, v22, v23, (__int64)v19, 0, 0, v46, v47, v48, v49, v50, v51);
  }
  else
  {
    v40 &= ~1u;
    v39 = 0;
    LODWORD(v41) = 0;
    sub_12843D0(v24, (_DWORD *)(v7 + 36), v20, v21, v22, v23, (__int64)v19, 0, 0, v46, v47, v48, v49, v50, v51);
  }
  result = 0;
  if ( (*(_BYTE *)(a2 + 25) & 4) == 0 )
  {
    v54 = v48;
    LODWORD(v46) = v35;
    v27 = *a1;
    v53 = v47;
    v55 = v49;
    v52 = v46;
    v56 = v50;
    v57 = v51;
    sub_1287CD0((__int64)v38, v27, (_DWORD *)(v7 + 36), v25, v50, v51, v46, v47, v48, v49, v50, v51);
    return (_QWORD *)v38[0];
  }
  return result;
}
