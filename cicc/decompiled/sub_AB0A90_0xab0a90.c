// Function: sub_AB0A90
// Address: 0xab0a90
//
__int64 __fastcall sub_AB0A90(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v4; // eax
  __int64 v5; // rdx
  unsigned __int64 v6; // rdx
  bool v7; // cc
  __int64 v8; // rdx
  unsigned int v9; // r13d
  int v10; // r14d
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // r14d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // [rsp+0h] [rbp-90h] BYREF
  unsigned int v20; // [rsp+8h] [rbp-88h]
  __int64 v21; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-78h]
  __int64 v23; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-68h]
  __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-58h]
  unsigned __int64 v27; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-48h]
  __int64 v29; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-38h]

  if ( sub_AAF7D0(a2) )
  {
    v2 = *(_DWORD *)(a2 + 8);
    sub_9691E0(a1, v2, 0, 0, 0);
    sub_9691E0(a1 + 16, v2, 0, 0, 0);
    return a1;
  }
  sub_AB0A00((__int64)&v19, a2);
  sub_AB0910((__int64)&v21, a2);
  v4 = v20;
  v26 = v20;
  if ( v20 <= 0x40 )
  {
    v25 = v19;
LABEL_6:
    v5 = v19;
    goto LABEL_7;
  }
  sub_C43780(&v25, &v19);
  v4 = v20;
  v24 = v20;
  if ( v20 <= 0x40 )
    goto LABEL_6;
  sub_C43780(&v23, &v19);
  v4 = v24;
  if ( v24 > 0x40 )
  {
    sub_C43D10(&v23, &v19, v16, v17, v18);
    v4 = v24;
    v6 = v23;
    goto LABEL_9;
  }
  v5 = v23;
LABEL_7:
  v6 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & ~v5;
  if ( !v4 )
    v6 = 0;
LABEL_9:
  v28 = v4;
  v27 = v6;
  v30 = v26;
  v29 = v25;
  v23 = sub_C4DDA0(&v19, &v21);
  if ( !BYTE4(v23) )
    goto LABEL_10;
  v8 = v28;
  v9 = v23 + 1;
  v26 = v28;
  v10 = v23 + 1 - v28;
  if ( v28 <= 0x40 )
  {
    v25 = 0;
    if ( v9 == v28 )
    {
      v12 = 0;
      goto LABEL_24;
    }
    v11 = v9;
    goto LABEL_19;
  }
  sub_C43690(&v25, 0, 0);
  v8 = v26;
  v11 = v10 + v26;
  if ( v26 != (_DWORD)v11 )
  {
LABEL_19:
    if ( (unsigned int)v11 > 0x3F || (unsigned int)v8 > 0x40 )
      sub_C43C90(&v25, v11, v8);
    else
      v25 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v10 + 64) << v11;
  }
  if ( v28 > 0x40 )
  {
    sub_C43B90(&v27, &v25);
    LODWORD(v8) = v26;
    goto LABEL_25;
  }
  v12 = v25;
  LODWORD(v8) = v26;
LABEL_24:
  v27 &= v12;
LABEL_25:
  if ( (unsigned int)v8 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  v13 = v30;
  v26 = v30;
  v14 = v9 - v30;
  if ( v30 <= 0x40 )
  {
    v25 = 0;
    if ( v9 == v30 )
    {
      v15 = 0;
      goto LABEL_35;
    }
    goto LABEL_30;
  }
  sub_C43690(&v25, 0, 0);
  v13 = v26;
  v9 = v14 + v26;
  if ( v26 != v14 + v26 )
  {
LABEL_30:
    if ( v9 > 0x3F || (unsigned int)v13 > 0x40 )
      sub_C43C90(&v25, v9, v13);
    else
      v25 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v14 + 64) << v9;
  }
  if ( v30 > 0x40 )
  {
    sub_C43B90(&v29, &v25);
    goto LABEL_36;
  }
  v15 = v25;
LABEL_35:
  v29 &= v15;
LABEL_36:
  sub_969240(&v25);
LABEL_10:
  v7 = v22 <= 0x40;
  *(_DWORD *)(a1 + 8) = v28;
  *(_QWORD *)a1 = v27;
  *(_DWORD *)(a1 + 24) = v30;
  *(_QWORD *)(a1 + 16) = v29;
  if ( !v7 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return a1;
}
