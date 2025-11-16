// Function: sub_1793A00
// Address: 0x1793a00
//
_QWORD *__fastcall sub_1793A00(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _QWORD *v5; // r12
  __int64 v7; // r14
  unsigned int v8; // r15d
  unsigned int v9; // eax
  __int64 **v11; // rdx
  int v12; // edi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned __int64 *v16; // r13
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 *v19; // rsi
  _QWORD *v20; // rdi
  __int64 v21; // rdx
  bool v22; // zf
  __int64 v23; // rsi
  _QWORD *v24; // r13
  __int64 v25; // r14
  __int64 v26; // rsi
  unsigned __int8 *v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rdi
  unsigned __int64 *v30; // r13
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rsi
  _QWORD *v36; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v37; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v38[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v39; // [rsp+30h] [rbp-60h]
  _QWORD v40[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v41; // [rsp+50h] [rbp-40h]

  v5 = (_QWORD *)a2;
  v7 = *(_QWORD *)a2;
  v8 = sub_16431D0(*(_QWORD *)a2);
  v9 = sub_16431D0(a3);
  if ( v8 >= v9 )
  {
    if ( v7 == a3 || v8 == v9 )
      return v5;
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      v11 = (__int64 **)a3;
      v12 = 36;
LABEL_8:
      v5 = (_QWORD *)sub_15A46C0(v12, (__int64 ***)a2, v11, 0);
      v13 = sub_14DBA30((__int64)v5, *(_QWORD *)(a1 + 96), 0);
      if ( v13 )
        return (_QWORD *)v13;
      return v5;
    }
    v41 = 257;
    v28 = sub_15FDBD0(36, a2, a3, (__int64)v40, 0);
    v29 = *(_QWORD *)(a1 + 8);
    v5 = (_QWORD *)v28;
    if ( v29 )
    {
      v30 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v29 + 40, v28);
      v31 = v5[3];
      v32 = *v30;
      v5[4] = v30;
      v32 &= 0xFFFFFFFFFFFFFFF8LL;
      v5[3] = v32 | v31 & 7;
      *(_QWORD *)(v32 + 8) = v5 + 3;
      *v30 = *v30 & 7 | (unsigned __int64)(v5 + 3);
    }
    v19 = a4;
    v20 = v5;
    sub_164B780((__int64)v5, a4);
    v22 = *(_QWORD *)(a1 + 80) == 0;
    v37 = v5;
    if ( !v22 )
    {
      (*(void (__fastcall **)(__int64, _QWORD **))(a1 + 88))(a1 + 64, &v37);
      v33 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        return v5;
      v24 = v38;
      v38[0] = *(_QWORD *)a1;
      v25 = (__int64)(v5 + 6);
      sub_1623A60((__int64)v38, v33, 2);
      v34 = v5[6];
      if ( v34 )
        sub_161E7C0((__int64)(v5 + 6), v34);
      v27 = (unsigned __int8 *)v38[0];
      v5[6] = v38[0];
      if ( !v27 )
        return v5;
LABEL_20:
      sub_1623210((__int64)v24, v27, v25);
      return v5;
    }
LABEL_29:
    sub_4263D6(v20, v19, v21);
  }
  if ( v7 == a3 )
    return v5;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
  {
    v11 = (__int64 **)a3;
    v12 = 37;
    goto LABEL_8;
  }
  v39 = 257;
  v14 = sub_15FDBD0(37, a2, a3, (__int64)v38, 0);
  v15 = *(_QWORD *)(a1 + 8);
  v5 = (_QWORD *)v14;
  if ( v15 )
  {
    v16 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v15 + 40, v14);
    v17 = v5[3];
    v18 = *v16;
    v5[4] = v16;
    v18 &= 0xFFFFFFFFFFFFFFF8LL;
    v5[3] = v18 | v17 & 7;
    *(_QWORD *)(v18 + 8) = v5 + 3;
    *v16 = *v16 & 7 | (unsigned __int64)(v5 + 3);
  }
  v19 = a4;
  v20 = v5;
  sub_164B780((__int64)v5, a4);
  v22 = *(_QWORD *)(a1 + 80) == 0;
  v36 = v5;
  if ( v22 )
    goto LABEL_29;
  (*(void (__fastcall **)(__int64, _QWORD **))(a1 + 88))(a1 + 64, &v36);
  v23 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v24 = v40;
    v40[0] = *(_QWORD *)a1;
    v25 = (__int64)(v5 + 6);
    sub_1623A60((__int64)v40, v23, 2);
    v26 = v5[6];
    if ( v26 )
      sub_161E7C0((__int64)(v5 + 6), v26);
    v27 = (unsigned __int8 *)v40[0];
    v5[6] = v40[0];
    if ( v27 )
      goto LABEL_20;
  }
  return v5;
}
