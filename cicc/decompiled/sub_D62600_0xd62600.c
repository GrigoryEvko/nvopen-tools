// Function: sub_D62600
// Address: 0xd62600
//
__int64 __fastcall sub_D62600(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // r13d
  unsigned __int8 *v7; // r10
  char v8; // al
  unsigned int v9; // eax
  unsigned __int8 *v10; // r10
  __int64 v11; // rdi
  unsigned int v12; // eax
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v16; // rdi
  unsigned __int8 v17; // dl
  __int64 v18; // rsi
  unsigned int v19; // r13d
  unsigned __int8 *v20; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v21; // [rsp+8h] [rbp-88h]
  char v22; // [rsp+1Fh] [rbp-71h] BYREF
  __int64 v23; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-68h]
  __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-58h]
  __int64 v27; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-48h]
  __int64 v29; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-38h]

  v24 = sub_AE43F0(*(_QWORD *)a2, *(_QWORD *)(a3 + 8));
  v6 = v24;
  if ( v24 > 0x40 )
    sub_C43690((__int64)&v23, 0, 0);
  else
    v23 = 0;
  v7 = sub_BD45C0((unsigned __int8 *)a3, *(_QWORD *)a2, (__int64)&v23, 1, 1, 0, 0, 0);
  v8 = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)(v8 - 2) > 1u )
    goto LABEL_4;
  v17 = *v7;
  if ( *v7 <= 0x1Cu )
  {
    if ( v17 != 5 || *((_WORD *)v7 + 1) != 34 )
      goto LABEL_4;
  }
  else if ( v17 != 63 )
  {
    goto LABEL_4;
  }
  v18 = *(_QWORD *)a2;
  LOBYTE(v27) = (v8 == 2) + 2;
  v7 = sub_BD45C0(
         v7,
         v18,
         (__int64)&v23,
         1,
         1,
         0,
         (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_D5C6B0,
         (__int64)&v27);
LABEL_4:
  v20 = v7;
  v9 = sub_AE43F0(*(_QWORD *)a2, *((_QWORD *)v7 + 1));
  v10 = v20;
  *(_DWORD *)(a2 + 32) = v9;
  v28 = v9;
  if ( v9 > 0x40 )
  {
    sub_C43690((__int64)&v27, 0, 0);
    v10 = v20;
  }
  else
  {
    v27 = 0;
  }
  if ( *(_DWORD *)(a2 + 48) > 0x40u )
  {
    v11 = *(_QWORD *)(a2 + 40);
    if ( v11 )
    {
      v21 = v10;
      j_j___libc_free_0_0(v11);
      v10 = v21;
    }
  }
  *(_QWORD *)(a2 + 40) = v27;
  *(_DWORD *)(a2 + 48) = v28;
  sub_D61E90((__int64)&v27, a2, v10);
  if ( *(_DWORD *)(a2 + 32) == v6 )
  {
    v19 = v24;
    if ( v24 <= 0x40 )
    {
      if ( v23 )
        goto LABEL_12;
    }
    else if ( v19 != (unsigned int)sub_C444A0((__int64)&v23) )
    {
      goto LABEL_12;
    }
    *(_DWORD *)(a1 + 8) = v28;
    *(_QWORD *)a1 = v27;
    *(_DWORD *)(a1 + 24) = v30;
    *(_QWORD *)(a1 + 16) = v29;
    goto LABEL_15;
  }
  if ( v28 > 1 && !(unsigned __int8)sub_D5C0A0(&v27, v6) )
  {
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    v27 = 0;
    v28 = 1;
  }
  if ( v30 > 1 && !(unsigned __int8)sub_D5C0A0(&v29, v6) )
  {
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    v12 = v28;
    v29 = 0;
    v13 = 1;
    v30 = 1;
    if ( v28 <= 1 )
      goto LABEL_14;
    goto LABEL_24;
  }
LABEL_12:
  v12 = v28;
  if ( v28 <= 1 )
  {
    v13 = v30;
    if ( v30 <= 1 )
    {
LABEL_14:
      *(_DWORD *)(a1 + 8) = v12;
      v14 = v27;
      *(_DWORD *)(a1 + 24) = v13;
      *(_QWORD *)a1 = v14;
      *(_QWORD *)(a1 + 16) = v29;
      goto LABEL_15;
    }
    goto LABEL_29;
  }
LABEL_24:
  sub_C45F70((__int64)&v25, (__int64)&v27, (__int64)&v23, &v22);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  v27 = v25;
  v28 = v26;
  if ( v22 )
  {
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    v27 = 0;
    v28 = 1;
  }
  v13 = v30;
  if ( v30 > 1 )
  {
LABEL_29:
    sub_C46BD0((__int64)&v25, (__int64)&v29, (__int64)&v23, &v22);
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    v13 = v26;
    v29 = v25;
    v30 = v26;
    if ( v22 )
    {
      if ( v26 > 0x40 && v25 )
        j_j___libc_free_0_0(v25);
      v29 = 0;
      v13 = 1;
      v30 = 1;
    }
  }
  v12 = v28;
  if ( v28 <= 1 )
    goto LABEL_14;
  v16 = v27;
  if ( v28 > 0x40 )
    v16 = *(_QWORD *)(v27 + 8LL * ((v28 - 1) >> 6));
  if ( (v16 & (1LL << ((unsigned __int8)v28 - 1))) == 0 || (unsigned __int8)(*(_BYTE *)(a2 + 16) - 2) > 1u )
    goto LABEL_14;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 8) = 1;
  *(_DWORD *)(a1 + 24) = 1;
  if ( v13 > 0x40 && v29 )
  {
    j_j___libc_free_0_0(v29);
    v12 = v28;
  }
  if ( v12 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
LABEL_15:
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  return a1;
}
