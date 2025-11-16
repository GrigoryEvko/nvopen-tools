// Function: sub_3422210
// Address: 0x3422210
//
char __fastcall sub_3422210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  unsigned __int16 v8; // dx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // eax
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // r8
  unsigned int v16; // edx
  __int64 *v17; // rbx
  char result; // al
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rax
  unsigned int v22; // [rsp+0h] [rbp-90h]
  char v24; // [rsp+8h] [rbp-88h]
  char v25; // [rsp+8h] [rbp-88h]
  unsigned __int64 v26; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-78h]
  unsigned __int64 v28; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-68h]
  unsigned __int64 v30; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-58h]
  unsigned __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  __int64 v33; // [rsp+48h] [rbp-48h]
  __int64 v34; // [rsp+50h] [rbp-40h]
  __int64 v35; // [rsp+58h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v8 = *(_WORD *)v7;
  v9 = *(_QWORD *)(v7 + 8);
  v10 = *(_QWORD *)(a4 + 96);
  LOWORD(v32) = v8;
  v33 = v9;
  if ( v8 )
  {
    if ( v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
      BUG();
    v12 = 16LL * (v8 - 1);
    v11 = *(_QWORD *)&byte_444C4A0[v12];
    LOBYTE(v12) = byte_444C4A0[v12 + 8];
  }
  else
  {
    v11 = sub_3007260((__int64)&v32);
    v34 = v11;
    v35 = v12;
  }
  v32 = v11;
  LOBYTE(v33) = v12;
  v13 = sub_CA1930(&v32);
  v27 = v13;
  if ( v13 > 0x40 )
  {
    sub_C43690((__int64)&v26, a5, 0);
  }
  else
  {
    v14 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v13) & a5;
    v15 = 0;
    if ( v13 )
      v15 = v14;
    v26 = v15;
  }
  v16 = *(_DWORD *)(v10 + 32);
  if ( v16 <= 0x40 )
  {
    v19 = *(_QWORD *)(v10 + 24);
    v20 = v26;
    result = 1;
    if ( v19 == v26 )
      goto LABEL_10;
    if ( (v19 & ~v26) != 0 )
    {
      result = 0;
      goto LABEL_10;
    }
    goto LABEL_16;
  }
  v17 = (__int64 *)(v10 + 24);
  v22 = v16;
  result = sub_C43C50((__int64)v17, (const void **)&v26);
  if ( result )
    goto LABEL_10;
  result = sub_C446F0(v17, (__int64 *)&v26);
  if ( !result )
    goto LABEL_10;
  v31 = v22;
  sub_C43780((__int64)&v30, (const void **)v17);
  v16 = v31;
  if ( v31 <= 0x40 )
  {
    v19 = v30;
    v20 = v26;
LABEL_16:
    v21 = 0;
    if ( v16 )
      v21 = v20 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v19;
    goto LABEL_18;
  }
  sub_C43D10((__int64)&v30);
  v16 = v31;
  v31 = 0;
  LODWORD(v33) = v16;
  v32 = v30;
  if ( v16 > 0x40 )
  {
    sub_C43B90(&v32, (__int64 *)&v26);
    v29 = v33;
    v28 = v32;
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    goto LABEL_19;
  }
  v21 = v26 & v30;
LABEL_18:
  v29 = v16;
  v28 = v21;
LABEL_19:
  result = sub_33DD210(*(_QWORD *)(a1 + 64), a2, a3, (__int64)&v28, 0);
  if ( v29 > 0x40 && v28 )
  {
    v25 = result;
    j_j___libc_free_0_0(v28);
    result = v25;
  }
LABEL_10:
  if ( v27 > 0x40 )
  {
    if ( v26 )
    {
      v24 = result;
      j_j___libc_free_0_0(v26);
      return v24;
    }
  }
  return result;
}
