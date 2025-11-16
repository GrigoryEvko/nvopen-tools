// Function: sub_1758F20
// Address: 0x1758f20
//
_QWORD *__fastcall sub_1758F20(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _BYTE *v6; // r13
  __int64 v7; // rdx
  char v8; // al
  __int64 v9; // r14
  _QWORD *result; // rax
  _BYTE *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rbx
  unsigned __int8 *v15; // r12
  __int64 v16; // rax
  __int64 v17; // r13
  unsigned __int8 *v18; // rax
  __int64 v19; // rax
  unsigned __int64 *v20; // r14
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  __int16 v27; // [rsp-94h] [rbp-94h]
  __int64 v28; // [rsp-90h] [rbp-90h]
  _QWORD *v29; // [rsp-90h] [rbp-90h]
  unsigned __int8 *v30; // [rsp-80h] [rbp-80h] BYREF
  __int64 v31[2]; // [rsp-78h] [rbp-78h] BYREF
  __int16 v32; // [rsp-68h] [rbp-68h]
  __int64 v33; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v34; // [rsp-50h] [rbp-50h]
  __int16 v35; // [rsp-48h] [rbp-48h]

  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) != 11 )
    return 0;
  v6 = *(_BYTE **)(a3 - 24);
  v7 = *(_QWORD *)v6;
  v8 = *(_BYTE *)(*(_QWORD *)v6 + 8LL);
  if ( v8 == 16 )
  {
    if ( *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL) == 11 )
      goto LABEL_4;
    return 0;
  }
  if ( v8 != 11 )
    return 0;
LABEL_4:
  if ( v6[16] != 85 )
    return 0;
  v9 = *((_QWORD *)v6 - 9);
  if ( !v9 )
    return 0;
  if ( *(_BYTE *)(*((_QWORD *)v6 - 6) + 16LL) != 9 )
    return 0;
  v11 = (_BYTE *)*((_QWORD *)v6 - 3);
  if ( v11[16] > 0x10u )
    return 0;
  v12 = sub_15A1020(v11, a2, v7, (__int64)a4);
  if ( !v12 )
    return 0;
  v28 = v12;
  if ( *(_BYTE *)(v12 + 16) != 13 )
    return 0;
  v13 = *(_QWORD *)(*(_QWORD *)v6 + 24LL);
  v27 = *(_WORD *)(a2 + 18);
  if ( !sub_16A8E60((__int64)a4, *(_DWORD *)(v13 + 8) >> 8) )
    return 0;
  v14 = *(_QWORD *)(a1 + 8);
  v32 = 257;
  if ( *(_BYTE *)(v9 + 16) > 0x10u || *(_BYTE *)(v28 + 16) > 0x10u )
  {
    v35 = 257;
    v18 = (unsigned __int8 *)sub_1648A60(56, 2u);
    v15 = v18;
    if ( v18 )
      sub_15FA320((__int64)v18, (_QWORD *)v9, v28, (__int64)&v33, 0);
    v19 = *(_QWORD *)(v14 + 8);
    if ( v19 )
    {
      v20 = *(unsigned __int64 **)(v14 + 16);
      sub_157E9D0(v19 + 40, (__int64)v15);
      v21 = *((_QWORD *)v15 + 3);
      v22 = *v20;
      *((_QWORD *)v15 + 4) = v20;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v15 + 3) = v22 | v21 & 7;
      *(_QWORD *)(v22 + 8) = v15 + 24;
      *v20 = *v20 & 7 | (unsigned __int64)(v15 + 24);
    }
    sub_164B780((__int64)v15, v31);
    v30 = v15;
    if ( !*(_QWORD *)(v14 + 80) )
      sub_4263D6(v15, v31, v23);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(v14 + 88))(v14 + 64, &v30);
    v24 = *(_QWORD *)v14;
    if ( *(_QWORD *)v14 )
    {
      v30 = *(unsigned __int8 **)v14;
      sub_1623A60((__int64)&v30, v24, 2);
      v25 = *((_QWORD *)v15 + 6);
      if ( v25 )
        sub_161E7C0((__int64)(v15 + 48), v25);
      v26 = v30;
      *((_QWORD *)v15 + 6) = v30;
      if ( v26 )
        sub_1623210((__int64)&v30, v26, (__int64)(v15 + 48));
    }
  }
  else
  {
    v15 = (unsigned __int8 *)sub_15A37D0((_BYTE *)v9, v28, 0);
    v16 = sub_14DBA30((__int64)v15, *(_QWORD *)(v14 + 96), 0);
    if ( v16 )
      v15 = (unsigned __int8 *)v16;
  }
  sub_16A5A50((__int64)&v33, a4, *(_DWORD *)(v13 + 8) >> 8);
  v17 = sub_15A1070(v13, (__int64)&v33);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  v35 = 257;
  result = sub_1648A60(56, 2u);
  if ( result )
  {
    v29 = result;
    sub_17582E0((__int64)result, v27 & 0x7FFF, (__int64)v15, v17, (__int64)&v33);
    return v29;
  }
  return result;
}
