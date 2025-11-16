// Function: sub_13CD2E0
// Address: 0x13cd2e0
//
_QWORD *__fastcall sub_13CD2E0(_QWORD *a1, __int64 a2, char a3)
{
  _QWORD *v4; // r12
  _BYTE *v5; // rdi
  unsigned __int8 v6; // al
  _BYTE *v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 v9; // al
  _BYTE *v10; // r8
  char v11; // r13
  char v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // [rsp+8h] [rbp-98h]
  _BYTE *v17; // [rsp+8h] [rbp-98h]
  __int64 v18; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-88h]
  __int64 v20; // [rsp+20h] [rbp-80h]
  unsigned int v21; // [rsp+28h] [rbp-78h]
  __int64 v22; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-68h]
  __int64 v24; // [rsp+40h] [rbp-60h]
  unsigned int v25; // [rsp+48h] [rbp-58h]
  __int64 v26; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v27; // [rsp+58h] [rbp-48h]
  __int64 v28; // [rsp+60h] [rbp-40h]
  unsigned int v29; // [rsp+68h] [rbp-38h]

  v4 = a1;
  v5 = (_BYTE *)*(a1 - 3);
  v6 = v5[16];
  v7 = v5 + 24;
  if ( v6 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) != 16 )
      return 0;
    if ( v6 > 0x10u )
      return 0;
    v14 = sub_15A1020(v5);
    if ( !v14 || *(_BYTE *)(v14 + 16) != 13 )
      return 0;
    v7 = (_BYTE *)(v14 + 24);
  }
  v8 = *(_BYTE **)(a2 - 24);
  v9 = v8[16];
  v10 = v8 + 24;
  if ( v9 == 13 )
    goto LABEL_3;
  if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 16 )
    return 0;
  if ( v9 > 0x10u )
    return 0;
  v17 = v7;
  v15 = sub_15A1020(v8);
  if ( !v15 || *(_BYTE *)(v15 + 16) != 13 )
    return 0;
  v7 = v17;
  v10 = (_BYTE *)(v15 + 24);
LABEL_3:
  v16 = v10;
  sub_158B890(&v18, *((_WORD *)v4 + 9) & 0x7FFF, v7);
  sub_158B890(&v22, *(_WORD *)(a2 + 18) & 0x7FFF, v16);
  if ( a3 )
  {
    sub_158BE00(&v26, &v18, &v22);
    v13 = sub_158A120(&v26);
    if ( v29 > 0x40 && v28 )
      j_j___libc_free_0_0(v28);
    if ( v27 > 0x40 && v26 )
      j_j___libc_free_0_0(v26);
    if ( v13 )
    {
      v4 = (_QWORD *)sub_15A0640(*v4);
      goto LABEL_12;
    }
    if ( (unsigned __int8)sub_158BB40(&v18, &v22) )
      goto LABEL_49;
    if ( (unsigned __int8)sub_158BB40(&v22, &v18) )
      goto LABEL_12;
    goto LABEL_43;
  }
  sub_158C3A0(&v26, &v18, &v22);
  v11 = sub_158A0B0(&v26);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v11 )
  {
    v4 = (_QWORD *)sub_15A0600(*v4);
    goto LABEL_12;
  }
  if ( !(unsigned __int8)sub_158BB40(&v18, &v22) )
  {
    if ( (unsigned __int8)sub_158BB40(&v22, &v18) )
    {
LABEL_49:
      v4 = (_QWORD *)a2;
      goto LABEL_12;
    }
LABEL_43:
    v4 = 0;
  }
LABEL_12:
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return v4;
}
