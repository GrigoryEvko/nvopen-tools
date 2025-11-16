// Function: sub_AAD630
// Address: 0xaad630
//
__int64 __fastcall sub_AAD630(__int64 a1, int *a2, __int64 a3, __int64 *a4, _QWORD *a5)
{
  unsigned int v6; // ebx
  _QWORD *v7; // r9
  __int64 v8; // r12
  __int64 v9; // r12
  __int64 v10; // r12
  char v11; // cl
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rbx
  char v14; // r12
  int v15; // eax
  unsigned int v16; // r14d
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rax
  int v20; // eax
  _QWORD *v22; // [rsp+0h] [rbp-70h]
  char v23; // [rsp+8h] [rbp-68h]
  _QWORD *v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-38h]

  v6 = *((_DWORD *)a4 + 2);
  v7 = (_QWORD *)*((_QWORD *)a2 + 1);
  v26 = v6;
  if ( v6 <= 0x40 )
  {
    v8 = *a4;
LABEL_3:
    v9 = *a5 & v8;
    v28 = v6;
    v25 = v9;
    v27 = v9;
    v26 = 0;
LABEL_4:
    v10 = *v7 | v9;
    v28 = 0;
    v27 = v10;
LABEL_5:
    if ( v6 )
    {
      v11 = 64 - v6;
      v6 = 64;
      v12 = ~(v10 << v11);
      if ( v12 )
      {
        _BitScanReverse64(&v13, v12);
        v6 = v13 ^ 0x3F;
      }
    }
    goto LABEL_8;
  }
  v22 = a5;
  v24 = v7;
  sub_C43780(&v25, a4);
  v6 = v26;
  v7 = v24;
  a5 = v22;
  if ( v26 <= 0x40 )
  {
    v8 = v25;
    goto LABEL_3;
  }
  sub_C43B90(&v25, v22);
  v6 = v26;
  v9 = v25;
  v26 = 0;
  v7 = v24;
  v28 = v6;
  v27 = v25;
  if ( v6 <= 0x40 )
    goto LABEL_4;
  sub_C43BD0(&v27, v24);
  v6 = v28;
  v10 = v27;
  v28 = 0;
  v30 = v6;
  v29 = v27;
  if ( v6 <= 0x40 )
    goto LABEL_5;
  v6 = sub_C44500(&v29);
  if ( v10 )
  {
    j_j___libc_free_0_0(v10);
    if ( v28 > 0x40 )
    {
      if ( v27 )
        j_j___libc_free_0_0(v27);
    }
  }
LABEL_8:
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  v15 = *a2;
  v30 = *(_DWORD *)(a3 + 8);
  v14 = v30;
  v16 = v6 + v30 - v15;
  if ( v30 <= 0x40 )
  {
    v29 = 0;
    v17 = v30;
    v18 = v15 - v6;
    if ( v30 == (_DWORD)v18 )
    {
      v19 = 0;
      goto LABEL_18;
    }
    if ( (unsigned int)v18 > 0x3F )
      goto LABEL_25;
    goto LABEL_14;
  }
  v23 = v15;
  sub_C43690(&v29, 0, 0);
  v17 = v30;
  LOBYTE(v15) = v23;
  v18 = v30 - v16;
  if ( v30 != (_DWORD)v18 )
  {
    if ( (unsigned int)v18 > 0x3F )
      goto LABEL_25;
LABEL_14:
    if ( (unsigned int)v17 <= 0x40 )
    {
      v29 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v15 - v14 + 64 - (unsigned __int8)v6) << v18;
      goto LABEL_16;
    }
LABEL_25:
    sub_C43C90(&v29, v18, v17);
    if ( *(_DWORD *)(a3 + 8) <= 0x40u )
      goto LABEL_17;
LABEL_26:
    sub_C43B90(a3, &v29);
    LODWORD(v18) = v30;
    goto LABEL_19;
  }
LABEL_16:
  if ( *(_DWORD *)(a3 + 8) > 0x40u )
    goto LABEL_26;
LABEL_17:
  v19 = v29;
  LODWORD(v18) = v30;
LABEL_18:
  *(_QWORD *)a3 &= v19;
LABEL_19:
  if ( (unsigned int)v18 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  v20 = *(_DWORD *)(a3 + 8);
  *(_DWORD *)(a3 + 8) = 0;
  *(_DWORD *)(a1 + 8) = v20;
  *(_QWORD *)a1 = *(_QWORD *)a3;
  return a1;
}
