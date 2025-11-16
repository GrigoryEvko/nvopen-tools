// Function: sub_1D493F0
// Address: 0x1d493f0
//
char __fastcall sub_1D493F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // r14
  char v10; // di
  __int64 v11; // rax
  unsigned int v12; // eax
  unsigned __int64 v13; // r8
  unsigned int v14; // r15d
  __int64 *v15; // r14
  char result; // al
  __int64 v17; // rdi
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // rax
  char v21; // [rsp+8h] [rbp-78h]
  char v22; // [rsp+8h] [rbp-78h]
  unsigned __int64 v23; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-48h]
  __int64 v29; // [rsp+40h] [rbp-40h] BYREF
  __int64 v30; // [rsp+48h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v9 = *(_QWORD *)(a4 + 88);
  v10 = *(_BYTE *)v8;
  v11 = *(_QWORD *)(v8 + 8);
  LOBYTE(v29) = v10;
  v30 = v11;
  if ( v10 )
    v12 = sub_1D46910(v10);
  else
    v12 = sub_1F58D40(&v29, a2, a3, a4, a5, a6);
  v24 = v12;
  if ( v12 > 0x40 )
  {
    sub_16A4EF0((__int64)&v23, a5, 0);
    v14 = *(_DWORD *)(v9 + 32);
    if ( v14 > 0x40 )
      goto LABEL_5;
LABEL_12:
    v17 = *(_QWORD *)(v9 + 24);
    v18 = v23;
    result = 1;
    if ( v17 == v23 )
      goto LABEL_7;
    if ( (v17 & ~v23) != 0 )
    {
      result = 0;
      goto LABEL_7;
    }
    goto LABEL_14;
  }
  v13 = a5 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v12);
  v14 = *(_DWORD *)(v9 + 32);
  v23 = v13;
  if ( v14 <= 0x40 )
    goto LABEL_12;
LABEL_5:
  v15 = (__int64 *)(v9 + 24);
  result = sub_16A5220((__int64)v15, (const void **)&v23);
  if ( result )
    goto LABEL_7;
  result = sub_16A5A00(v15, (__int64 *)&v23);
  if ( !result )
    goto LABEL_7;
  v28 = v14;
  sub_16A4FD0((__int64)&v27, (const void **)v15);
  v14 = v28;
  if ( v28 <= 0x40 )
  {
    v17 = v27;
    v18 = v23;
LABEL_14:
    v19 = ~v17 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
    goto LABEL_15;
  }
  sub_16A8F40(&v27);
  v14 = v28;
  v19 = v27;
  v28 = 0;
  LODWORD(v30) = v14;
  v29 = v27;
  if ( v14 > 0x40 )
  {
    sub_16A8890(&v29, (__int64 *)&v23);
    v26 = v30;
    v25 = v29;
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    goto LABEL_16;
  }
  v18 = v23;
LABEL_15:
  v26 = v14;
  v25 = v18 & v19;
LABEL_16:
  result = sub_1D1F940(*(_QWORD *)(a1 + 272), a2, a3, (__int64)&v25, 0);
  if ( v26 > 0x40 && v25 )
  {
    v22 = result;
    j_j___libc_free_0_0(v25);
    result = v22;
  }
LABEL_7:
  if ( v24 > 0x40 )
  {
    if ( v23 )
    {
      v21 = result;
      j_j___libc_free_0_0(v23);
      return v21;
    }
  }
  return result;
}
