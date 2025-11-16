// Function: sub_1196190
// Address: 0x1196190
//
unsigned __int8 *__fastcall sub_1196190(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 *result; // rax
  _BYTE *v4; // rax
  unsigned __int8 *v5; // r14
  _BYTE *v6; // rdx
  char v7; // al
  _BYTE *v9; // rdi
  _BYTE *v10; // rax
  _BYTE *v11; // rdx
  char v12; // al
  _BYTE *v13; // r15
  _BYTE *v14; // rax
  int v15; // eax
  unsigned __int8 *v16; // r15
  __int64 *v17; // rdx
  __int64 v18; // rbx
  _BYTE *v19; // rdx
  char v20; // al
  _BYTE *v21; // rdi
  _BYTE *v22; // rax
  int v23; // eax
  int v24; // edi
  __int64 v25; // rax
  unsigned __int8 *v26; // r14
  __int64 *v27; // rbx
  __int64 v28; // r13
  __int64 v29; // rbx
  __int64 v30; // rdx
  unsigned int v31; // esi
  _BYTE *v32; // rax
  _BYTE *v33; // rax
  _BYTE *v34; // rax
  _BYTE *v35; // rcx
  _BYTE *v36; // rax
  _BYTE *v37; // rax
  _BYTE *v38; // [rsp-70h] [rbp-70h]
  __int64 v39; // [rsp-70h] [rbp-70h]
  unsigned __int8 *v40; // [rsp-70h] [rbp-70h]
  _BYTE v41[32]; // [rsp-68h] [rbp-68h] BYREF
  __int16 v42; // [rsp-48h] [rbp-48h]

  if ( *a2 != 56 )
    return 0;
  v4 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( *v4 != 54 )
    return 0;
  v5 = (unsigned __int8 *)*((_QWORD *)v4 - 8);
  if ( *v5 <= 0x1Cu )
    return 0;
  v6 = (_BYTE *)*((_QWORD *)v4 - 4);
  v7 = *v6;
  if ( *v6 <= 0x1Cu )
    return 0;
  if ( v7 == 68 )
  {
    v34 = (_BYTE *)*((_QWORD *)v6 - 4);
    if ( *v34 != 44 )
      return 0;
    v9 = (_BYTE *)*((_QWORD *)v34 - 8);
    if ( *v9 > 0x15u )
      return 0;
    v10 = (_BYTE *)*((_QWORD *)v34 - 4);
    v38 = v10;
    if ( *v10 != 68 )
      goto LABEL_12;
LABEL_56:
    v35 = v10;
    v36 = (_BYTE *)*((_QWORD *)v10 - 4);
    if ( !v36 )
      v36 = v35;
    v38 = v36;
    goto LABEL_12;
  }
  if ( v7 != 44 )
    return 0;
  v9 = (_BYTE *)*((_QWORD *)v6 - 8);
  if ( *v9 > 0x15u )
    return 0;
  v10 = (_BYTE *)*((_QWORD *)v6 - 4);
  v38 = v10;
  if ( *v10 == 68 )
    goto LABEL_56;
LABEL_12:
  v11 = (_BYTE *)*((_QWORD *)a2 - 4);
  v12 = *v11;
  if ( *v11 <= 0x1Cu )
    return 0;
  if ( v12 == 68 )
  {
    v37 = (_BYTE *)*((_QWORD *)v11 - 4);
    if ( *v37 != 44 )
      return 0;
    v13 = (_BYTE *)*((_QWORD *)v37 - 8);
    if ( *v13 > 0x15u )
      return 0;
    v14 = (_BYTE *)*((_QWORD *)v37 - 4);
    if ( *v14 != 68 )
      goto LABEL_17;
LABEL_62:
    if ( *((_BYTE **)v14 - 4) == v38 )
      goto LABEL_18;
    goto LABEL_17;
  }
  if ( v12 != 44 )
    return 0;
  v13 = (_BYTE *)*((_QWORD *)v11 - 8);
  if ( *v13 > 0x15u )
    return 0;
  v14 = (_BYTE *)*((_QWORD *)v11 - 4);
  if ( *v14 == 68 )
    goto LABEL_62;
LABEL_17:
  if ( v14 != v38 )
    return 0;
LABEL_18:
  if ( !(unsigned __int8)sub_1194570((__int64)v9, *((_QWORD *)a2 + 1))
    || !(unsigned __int8)sub_1194570((__int64)v13, *((_QWORD *)a2 + 1)) )
  {
    return 0;
  }
  v15 = *v5;
  if ( (_BYTE)v15 == 67 )
  {
    v16 = (unsigned __int8 *)*((_QWORD *)v5 - 4);
    v15 = *v16;
    if ( (unsigned __int8)v15 <= 0x1Cu )
      return 0;
  }
  else
  {
    v16 = v5;
  }
  if ( (unsigned int)(v15 - 55) > 1 )
    return 0;
  if ( (v16[7] & 0x40) != 0 )
  {
    v17 = (__int64 *)*((_QWORD *)v16 - 1);
    v18 = *v17;
    if ( !*v17 )
      return 0;
  }
  else
  {
    v17 = (__int64 *)&v16[-32 * (*((_DWORD *)v16 + 1) & 0x7FFFFFF)];
    v18 = *v17;
    if ( !*v17 )
      return 0;
  }
  v19 = (_BYTE *)v17[4];
  if ( !v19 )
    return 0;
  v20 = *v19;
  if ( *v19 <= 0x1Cu )
    return 0;
  if ( v20 == 68 )
  {
    v32 = (_BYTE *)*((_QWORD *)v19 - 4);
    if ( *v32 != 44 )
      return 0;
    v21 = (_BYTE *)*((_QWORD *)v32 - 8);
    if ( *v21 > 0x15u )
      return 0;
    v33 = (_BYTE *)*((_QWORD *)v32 - 4);
    if ( (*v33 != 68 || *((_BYTE **)v33 - 4) != v38) && v33 != v38 )
      return 0;
  }
  else
  {
    if ( v20 != 44 )
      return 0;
    v21 = (_BYTE *)*((_QWORD *)v19 - 8);
    if ( *v21 > 0x15u )
      return 0;
    v22 = (_BYTE *)*((_QWORD *)v19 - 4);
    if ( (*v22 != 68 || v38 != *((_BYTE **)v22 - 4)) && v38 != v22 )
      return 0;
  }
  v39 = (__int64)v19;
  if ( !(unsigned __int8)sub_1194570((__int64)v21, *((_QWORD *)v16 + 1)) )
    return 0;
  v23 = *a2;
  v24 = v23 - 29;
  if ( (_BYTE)v23 == *v16 )
    return sub_F162A0(a1, (__int64)a2, (__int64)v5);
  if ( v5 == v16 )
  {
    v42 = 257;
    v40 = (unsigned __int8 *)sub_B504D0(v24, v18, v39, (__int64)v41, 0, 0);
    sub_B45260(v40, (__int64)v5, 1);
    return v40;
  }
  v25 = *(_QWORD *)(*((_QWORD *)a2 - 8) + 16LL);
  if ( !v25 || *(_QWORD *)(v25 + 8) )
  {
    result = *(unsigned __int8 **)(*((_QWORD *)a2 - 4) + 16LL);
    if ( !result )
      return result;
    if ( *((_QWORD *)result + 1) )
      return 0;
  }
  v42 = 257;
  v26 = (unsigned __int8 *)sub_B504D0(v24, v18, v39, (__int64)v41, 0, 0);
  sub_B45260(v26, (__int64)v16, 1);
  v27 = *(__int64 **)(a1 + 32);
  v42 = 257;
  (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)v27[11] + 16LL))(
    v27[11],
    v26,
    v41,
    v27[7],
    v27[8]);
  v28 = *v27;
  v29 = *v27 + 16LL * *((unsigned int *)v27 + 2);
  while ( v29 != v28 )
  {
    v30 = *(_QWORD *)(v28 + 8);
    v31 = *(_DWORD *)v28;
    v28 += 16;
    sub_B99FD0((__int64)v26, v31, v30);
  }
  v42 = 257;
  return (unsigned __int8 *)sub_B52120((__int64)v26, *((_QWORD *)a2 + 1), (__int64)v41, 0, 0);
}
