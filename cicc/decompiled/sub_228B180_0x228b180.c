// Function: sub_228B180
// Address: 0x228b180
//
__int64 __fastcall sub_228B180(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  __int64 *v6; // rdx
  unsigned int v7; // edx
  unsigned int v8; // ebx
  const void *v9; // rsi
  __int64 v10; // rsi
  bool v11; // al
  bool v12; // al
  unsigned int v13; // edx
  unsigned int v15; // eax
  int v16; // eax
  unsigned int v17; // ecx
  unsigned int v18; // ebx
  int v19; // eax
  int v20; // eax
  __int64 *v21; // [rsp+0h] [rbp-70h]
  unsigned int v22; // [rsp+Ch] [rbp-64h]
  unsigned int v23; // [rsp+Ch] [rbp-64h]
  unsigned int v24; // [rsp+Ch] [rbp-64h]
  unsigned int v25; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v26; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-58h]
  __int64 *v28; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-48h]
  unsigned __int64 v30; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 8);
  v27 = v5;
  if ( v5 <= 0x40 )
  {
    v6 = *(__int64 **)a2;
    v29 = v5;
    v26 = (unsigned __int64)v6;
LABEL_3:
    v28 = *(__int64 **)a2;
    goto LABEL_4;
  }
  sub_C43780((__int64)&v26, (const void **)a2);
  v29 = *(_DWORD *)(a2 + 8);
  if ( v29 <= 0x40 )
    goto LABEL_3;
  sub_C43780((__int64)&v28, (const void **)a2);
LABEL_4:
  sub_C4C400(a2, a3, (__int64)&v26, (__int64)&v28);
  v7 = v29;
  if ( v29 > 0x40 )
  {
    v24 = v29;
    v16 = sub_C444A0((__int64)&v28);
    v7 = v24;
    if ( v24 - v16 <= 0x40 && !*v28 )
    {
LABEL_20:
      v15 = v27;
      v27 = 0;
      *(_DWORD *)(a1 + 8) = v15;
      *(_QWORD *)a1 = v26;
      if ( v7 <= 0x40 )
        goto LABEL_16;
      goto LABEL_21;
    }
  }
  else if ( !v28 )
  {
    goto LABEL_20;
  }
  v8 = *(_DWORD *)(a2 + 8);
  v9 = *(const void **)a2;
  if ( v8 <= 0x40 )
  {
    if ( !v8 )
      goto LABEL_11;
    v10 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v8)) >> (64 - (unsigned __int8)v8);
    goto LABEL_9;
  }
  v17 = v8 - 1;
  v18 = v8 + 1;
  v21 = *(__int64 **)a2;
  v25 = v7;
  if ( (*((_QWORD *)v9 + (v17 >> 6)) & (1LL << v17)) != 0 )
  {
    v20 = sub_C44500(a2);
    v7 = v25;
    if ( v18 - v20 > 0x40 )
      goto LABEL_11;
    goto LABEL_28;
  }
  v19 = sub_C444A0(a2);
  v7 = v25;
  if ( v18 - v19 <= 0x40 )
  {
LABEL_28:
    v10 = *v21;
LABEL_9:
    if ( v10 <= 0 )
      goto LABEL_11;
  }
  v22 = v7;
  v11 = sub_AAD930(a3, 0);
  v7 = v22;
  if ( v11 )
    goto LABEL_20;
LABEL_11:
  v23 = v7;
  if ( sub_986F30(a2, 0) )
  {
    v12 = sub_986F30(a3, 0);
    v7 = v23;
    if ( v12 )
      goto LABEL_20;
  }
  v31 = v27;
  if ( v27 > 0x40 )
    sub_C43780((__int64)&v30, (const void **)&v26);
  else
    v30 = v26;
  sub_C46F20((__int64)&v30, 1u);
  v13 = v29;
  *(_DWORD *)(a1 + 8) = v31;
  *(_QWORD *)a1 = v30;
  if ( v13 <= 0x40 )
    goto LABEL_16;
LABEL_21:
  if ( v28 )
    j_j___libc_free_0_0((unsigned __int64)v28);
LABEL_16:
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  return a1;
}
