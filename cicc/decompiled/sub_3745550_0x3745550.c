// Function: sub_3745550
// Address: 0x3745550
//
__int64 __fastcall sub_3745550(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  unsigned int v7; // ebx
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rdx
  unsigned __int16 v11; // si
  __int64 (*v12)(); // r9
  __int64 v13; // rdx
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // r15d
  _QWORD *v19; // rdi
  __int64 v20; // rax
  unsigned int v21; // r15d
  __int64 v22; // rdx
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 (*v25)(); // rax
  unsigned int v26; // ebx
  __int64 v27; // rax
  __int64 v28; // rdx
  char v29; // al
  unsigned int v30; // eax
  __int64 v31; // r8
  __int64 (*v32)(); // rax
  unsigned int v33; // [rsp+0h] [rbp-80h] BYREF
  __int64 v34; // [rsp+8h] [rbp-78h]
  __int64 v35; // [rsp+10h] [rbp-70h] BYREF
  char v36; // [rsp+18h] [rbp-68h]
  __int64 v37; // [rsp+20h] [rbp-60h]
  __int64 v38; // [rsp+28h] [rbp-58h]
  __int64 v39; // [rsp+30h] [rbp-50h]
  __int64 v40; // [rsp+38h] [rbp-48h]
  __int64 v41; // [rsp+40h] [rbp-40h] BYREF
  __int64 v42; // [rsp+48h] [rbp-38h]

  v5 = sub_3746830(a1, a3);
  if ( !v5 )
    return 0;
  v7 = v5;
  v8 = sub_2D5BAE0(a1[16], a1[14], *(__int64 **)(a2 + 8), 0);
  v34 = v9;
  v10 = *a1;
  v11 = v8;
  v33 = v8;
  v12 = *(__int64 (**)())(v10 + 64);
  if ( v12 != sub_3740EE0 )
  {
    v23 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, _QWORD))v12)(a1, v8, v8, 244, v7);
    if ( v23 )
      goto LABEL_20;
    v11 = v33;
  }
  if ( v11 )
  {
    if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
      goto LABEL_49;
    v24 = 16LL * (v11 - 1);
    v14 = *(_QWORD *)&byte_444C4A0[v24];
    v15 = byte_444C4A0[v24 + 8];
  }
  else
  {
    v37 = sub_3007260((__int64)&v33);
    v38 = v13;
    v14 = v37;
    v15 = v38;
  }
  v41 = v14;
  LOBYTE(v42) = v15;
  if ( (unsigned __int64)sub_CA1930(&v41) > 0x40 )
    return 0;
  if ( (_WORD)v33 )
  {
    if ( (_WORD)v33 == 1 || (unsigned __int16)(v33 - 504) <= 7u )
      goto LABEL_49;
    v17 = 16LL * ((unsigned __int16)v33 - 1);
    v16 = *(_QWORD *)&byte_444C4A0[v17];
    LOBYTE(v17) = byte_444C4A0[v17 + 8];
  }
  else
  {
    v16 = sub_3007260((__int64)&v33);
    v39 = v16;
    v40 = v17;
  }
  LOBYTE(v42) = v17;
  v41 = v16;
  v18 = sub_CA1930(&v41);
  v19 = (_QWORD *)sub_BD5C60(a2);
  switch ( v18 )
  {
    case 1u:
      v20 = 2;
      v21 = 2;
LABEL_25:
      v22 = a1[16];
      goto LABEL_26;
    case 2u:
      v20 = 3;
      v21 = 3;
      goto LABEL_25;
    case 4u:
      v20 = 4;
      v21 = 4;
      goto LABEL_25;
    case 8u:
      v20 = 5;
      v21 = 5;
      goto LABEL_25;
    case 0x10u:
      v20 = 6;
      v21 = 6;
      goto LABEL_25;
    case 0x20u:
      v20 = 7;
      v21 = 7;
      goto LABEL_25;
    case 0x40u:
      v20 = 8;
      v21 = 8;
      goto LABEL_25;
    case 0x80u:
      v20 = 9;
      v21 = 9;
      goto LABEL_25;
  }
  LODWORD(v20) = sub_3007020(v19, v18);
  v21 = v20;
  if ( !(_WORD)v20 )
    return 0;
  v22 = a1[16];
  v20 = (unsigned __int16)v20;
LABEL_26:
  if ( !*(_QWORD *)(v22 + 8 * v20 + 112) )
    return 0;
  v25 = *(__int64 (**)())(*a1 + 64);
  if ( v25 == sub_3740EE0 )
    return 0;
  v26 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, _QWORD))v25)(
          a1,
          (unsigned __int16)v33,
          v21,
          234,
          v7);
  if ( !v26 )
    return 0;
  if ( (_WORD)v33 )
  {
    if ( (_WORD)v33 != 1 && (unsigned __int16)(v33 - 504) > 7u )
    {
      v28 = 16LL * ((unsigned __int16)v33 - 1);
      v27 = *(_QWORD *)&byte_444C4A0[v28];
      LOBYTE(v28) = byte_444C4A0[v28 + 8];
      goto LABEL_31;
    }
LABEL_49:
    BUG();
  }
  v27 = sub_3007260((__int64)&v33);
  v41 = v27;
  v42 = v28;
LABEL_31:
  v36 = v28;
  v35 = v27;
  v29 = sub_CA1930(&v35);
  v30 = sub_3749CE0(a1, v21, 188, v26, 1LL << (v29 - 1), v21);
  v31 = v30;
  if ( !v30 )
    return 0;
  v32 = *(__int64 (**)())(*a1 + 64);
  if ( v32 == sub_3740EE0 )
    return 0;
  v23 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, __int64))v32)(
          a1,
          v21,
          (unsigned __int16)v33,
          234,
          v31);
  if ( !v23 )
    return 0;
LABEL_20:
  sub_3742B00((__int64)a1, (_BYTE *)a2, v23, 1);
  return 1;
}
