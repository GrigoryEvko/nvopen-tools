// Function: sub_34224C0
// Address: 0x34224c0
//
char __fastcall sub_34224C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  unsigned __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // eax
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // r8
  unsigned int v17; // r14d
  char result; // al
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // rsi
  __int64 *v24; // [rsp+0h] [rbp-A0h]
  char v25; // [rsp+8h] [rbp-98h]
  char v26; // [rsp+8h] [rbp-98h]
  char v27; // [rsp+8h] [rbp-98h]
  char v28; // [rsp+8h] [rbp-98h]
  unsigned __int64 v29; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v30; // [rsp+18h] [rbp-88h]
  unsigned __int64 v31; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v32; // [rsp+28h] [rbp-78h]
  unsigned __int64 v33; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-68h]
  __int64 v35; // [rsp+40h] [rbp-60h]
  __int64 v36; // [rsp+48h] [rbp-58h]
  __int64 v37; // [rsp+50h] [rbp-50h] BYREF
  __int64 v38; // [rsp+58h] [rbp-48h]
  unsigned __int64 v39; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v40; // [rsp+68h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v11 = *(_QWORD *)(a4 + 96);
  LOWORD(v37) = v9;
  v38 = v10;
  if ( v9 )
  {
    if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
      BUG();
    v13 = 16LL * (v9 - 1);
    v12 = *(_QWORD *)&byte_444C4A0[v13];
    LOBYTE(v13) = byte_444C4A0[v13 + 8];
  }
  else
  {
    v12 = sub_3007260((__int64)&v37);
    v35 = v12;
    v36 = v13;
  }
  v37 = v12;
  LOBYTE(v38) = v13;
  v14 = sub_CA1930(&v37);
  v30 = v14;
  if ( v14 > 0x40 )
  {
    sub_C43690((__int64)&v29, a5, 0);
  }
  else
  {
    v15 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & a5;
    v16 = 0;
    if ( v14 )
      v16 = v15;
    v29 = v16;
  }
  v17 = *(_DWORD *)(v11 + 32);
  if ( v17 <= 0x40 )
  {
    v19 = *(_QWORD *)(v11 + 24);
    v20 = v29;
    result = 1;
    if ( v19 == v29 )
      goto LABEL_10;
    if ( (v19 & ~v29) != 0 )
    {
      result = 0;
      goto LABEL_10;
    }
    goto LABEL_22;
  }
  v24 = (__int64 *)(v11 + 24);
  result = sub_C43C50(v11 + 24, (const void **)&v29);
  if ( result )
    goto LABEL_10;
  result = sub_C446F0(v24, (__int64 *)&v29);
  if ( !result )
    goto LABEL_10;
  v34 = v17;
  sub_C43780((__int64)&v33, (const void **)v24);
  v17 = v34;
  if ( v34 <= 0x40 )
  {
    v19 = v33;
    v20 = v29;
LABEL_22:
    v21 = v20 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v17) & ~v19;
    v22 = 0;
    if ( v17 )
      v22 = v21;
    goto LABEL_24;
  }
  sub_C43D10((__int64)&v33);
  v17 = v34;
  v34 = 0;
  LODWORD(v38) = v17;
  v37 = v33;
  if ( v17 <= 0x40 )
  {
    v22 = v29 & v33;
LABEL_24:
    v32 = v17;
    v23 = *(_QWORD *)(a1 + 64);
    v31 = v22;
    sub_33DD090((__int64)&v37, v23, a2, a3, 0);
    goto LABEL_25;
  }
  sub_C43B90(&v37, (__int64 *)&v29);
  v17 = v38;
  v22 = v37;
  v32 = v38;
  v31 = v37;
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  sub_33DD090((__int64)&v37, *(_QWORD *)(a1 + 64), a2, a3, 0);
  if ( v17 > 0x40 )
  {
    result = sub_C446F0((__int64 *)&v31, (__int64 *)&v39);
    if ( !result )
      goto LABEL_27;
    goto LABEL_43;
  }
LABEL_25:
  if ( (v22 & ~v39) != 0 )
  {
    result = 0;
    goto LABEL_27;
  }
LABEL_43:
  result = 1;
LABEL_27:
  if ( v40 > 0x40 && v39 )
  {
    v26 = result;
    j_j___libc_free_0_0(v39);
    result = v26;
  }
  if ( (unsigned int)v38 > 0x40 && v37 )
  {
    v27 = result;
    j_j___libc_free_0_0(v37);
    result = v27;
  }
  if ( v17 > 0x40 && v22 )
  {
    v28 = result;
    j_j___libc_free_0_0(v22);
    result = v28;
  }
LABEL_10:
  if ( v30 > 0x40 )
  {
    if ( v29 )
    {
      v25 = result;
      j_j___libc_free_0_0(v29);
      return v25;
    }
  }
  return result;
}
