// Function: sub_228AEF0
// Address: 0x228aef0
//
__int64 __fastcall sub_228AEF0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  __int64 *v6; // rdx
  unsigned int v7; // edx
  unsigned int v8; // ebx
  const void *v9; // rsi
  __int64 v10; // rsi
  bool v11; // al
  unsigned int v12; // edx
  unsigned int v14; // ecx
  unsigned int v15; // ebx
  int v16; // eax
  bool v17; // al
  bool v18; // al
  unsigned int v19; // eax
  int v20; // eax
  int v21; // eax
  __int64 *v22; // [rsp+0h] [rbp-70h]
  unsigned int v23; // [rsp+Ch] [rbp-64h]
  unsigned int v24; // [rsp+Ch] [rbp-64h]
  unsigned int v25; // [rsp+Ch] [rbp-64h]
  unsigned int v26; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v27; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v28; // [rsp+18h] [rbp-58h]
  __int64 *v29; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+28h] [rbp-48h]
  unsigned __int64 v31; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 8);
  v28 = v5;
  if ( v5 <= 0x40 )
  {
    v6 = *(__int64 **)a2;
    v30 = v5;
    v27 = (unsigned __int64)v6;
LABEL_3:
    v29 = *(__int64 **)a2;
    goto LABEL_4;
  }
  sub_C43780((__int64)&v27, (const void **)a2);
  v30 = *(_DWORD *)(a2 + 8);
  if ( v30 <= 0x40 )
    goto LABEL_3;
  sub_C43780((__int64)&v29, (const void **)a2);
LABEL_4:
  sub_C4C400(a2, a3, (__int64)&v27, (__int64)&v29);
  v7 = v30;
  if ( v30 > 0x40 )
  {
    v26 = v30;
    v20 = sub_C444A0((__int64)&v29);
    v7 = v26;
    if ( v26 - v20 <= 0x40 && !*v29 )
      goto LABEL_22;
  }
  else if ( !v29 )
  {
    goto LABEL_22;
  }
  v8 = *(_DWORD *)(a2 + 8);
  v9 = *(const void **)a2;
  if ( v8 <= 0x40 )
  {
    if ( !v8 )
      goto LABEL_20;
    v10 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v8)) >> (64 - (unsigned __int8)v8);
LABEL_9:
    if ( v10 > 0 )
      goto LABEL_10;
LABEL_20:
    v25 = v7;
    v17 = sub_986F30(a2, 0);
    v7 = v25;
    if ( v17 )
    {
      v18 = sub_986F30(a3, 0);
      v7 = v25;
      if ( v18 )
        goto LABEL_11;
    }
LABEL_22:
    v19 = v28;
    v28 = 0;
    *(_DWORD *)(a1 + 8) = v19;
    *(_QWORD *)a1 = v27;
    if ( v7 <= 0x40 )
      goto LABEL_14;
    goto LABEL_23;
  }
  v14 = v8 - 1;
  v15 = v8 + 1;
  v22 = *(__int64 **)a2;
  v24 = v7;
  if ( (*((_QWORD *)v9 + (v14 >> 6)) & (1LL << v14)) != 0 )
  {
    v16 = sub_C44500(a2);
    v7 = v24;
    if ( v15 - v16 > 0x40 )
      goto LABEL_20;
    goto LABEL_31;
  }
  v21 = sub_C444A0(a2);
  v7 = v24;
  if ( v15 - v21 <= 0x40 )
  {
LABEL_31:
    v10 = *v22;
    goto LABEL_9;
  }
LABEL_10:
  v23 = v7;
  v11 = sub_AAD930(a3, 0);
  v7 = v23;
  if ( !v11 )
    goto LABEL_20;
LABEL_11:
  v32 = v28;
  if ( v28 > 0x40 )
    sub_C43780((__int64)&v31, (const void **)&v27);
  else
    v31 = v27;
  sub_C46A40((__int64)&v31, 1);
  v12 = v30;
  *(_DWORD *)(a1 + 8) = v32;
  *(_QWORD *)a1 = v31;
  if ( v12 <= 0x40 )
    goto LABEL_14;
LABEL_23:
  if ( v29 )
    j_j___libc_free_0_0((unsigned __int64)v29);
LABEL_14:
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  return a1;
}
