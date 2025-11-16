// Function: sub_D5D7C0
// Address: 0xd5d7c0
//
__int64 __fastcall sub_D5D7C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r15
  char v7; // r14
  const void *v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v11; // rsi
  unsigned int v12; // eax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int v16; // eax
  const void *v17; // rdi
  unsigned __int64 v18; // rax
  unsigned __int16 v19; // cx
  unsigned int v20; // eax
  const void *v21; // rdx
  unsigned int v22; // eax
  unsigned int v23; // edx
  unsigned int v24; // eax
  __int16 v25; // cx
  unsigned __int64 v26; // rdx
  unsigned __int16 v27; // cx
  unsigned int v28; // eax
  const void *v29; // rdx
  unsigned int v30; // eax
  unsigned int v31; // eax
  __int64 v32; // [rsp+8h] [rbp-B8h]
  unsigned __int16 v33; // [rsp+8h] [rbp-B8h]
  unsigned __int16 v34; // [rsp+8h] [rbp-B8h]
  bool v35; // [rsp+1Fh] [rbp-A1h] BYREF
  const void *v36; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v37; // [rsp+28h] [rbp-98h]
  const void *v38; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v39; // [rsp+38h] [rbp-88h]
  const void *v40; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v41; // [rsp+48h] [rbp-78h]
  const void *v42; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v43; // [rsp+58h] [rbp-68h]
  const void *v44; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v45; // [rsp+68h] [rbp-58h]
  const void *v46; // [rsp+70h] [rbp-50h] BYREF
  __int64 v47; // [rsp+78h] [rbp-48h]
  char v48; // [rsp+80h] [rbp-40h]

  v6 = *(_QWORD *)a2;
  v32 = *(_QWORD *)(a3 + 72);
  v7 = sub_AE5020(*(_QWORD *)a2, v32);
  v8 = (const void *)sub_9208B0(v6, v32);
  v47 = v9;
  v46 = v8;
  if ( (_BYTE)v9 && *(_BYTE *)(a2 + 16) != 2 )
  {
LABEL_3:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  v11 = (((unsigned __int64)v8 + 7) >> 3) + (1LL << v7) - 1;
  v12 = *(_DWORD *)(a2 + 32);
  v13 = v11 >> v7 << v7;
  if ( v12 <= 0x3F )
  {
    v14 = 0;
    if ( v12 )
      v14 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12);
    if ( v13 > v14 )
      goto LABEL_3;
    v37 = *(_DWORD *)(a2 + 32);
LABEL_10:
    v36 = (const void *)v13;
    if ( (unsigned __int8)sub_B4CE70(a3) )
      goto LABEL_11;
    goto LABEL_28;
  }
  v37 = *(_DWORD *)(a2 + 32);
  if ( v12 == 64 )
    goto LABEL_10;
  sub_C43690((__int64)&v36, v13, 0);
  if ( (unsigned __int8)sub_B4CE70(a3) )
  {
LABEL_11:
    v15 = *(_QWORD *)(a3 - 32);
    if ( *(_BYTE *)v15 == 17 )
    {
      v16 = *(_DWORD *)(v15 + 32);
      LODWORD(v47) = v16;
      if ( v16 > 0x40 )
      {
        sub_C43780((__int64)&v46, (const void **)(v15 + 24));
        v16 = v47;
      }
      else
      {
        v46 = *(const void **)(v15 + 24);
      }
      v48 = 1;
    }
    else
    {
      v23 = *(unsigned __int8 *)(a2 + 16);
      if ( (unsigned __int8)(v23 - 2) > 1u || (sub_D5C200((__int64)&v46, (char *)v15, v23, 0), !v48) )
      {
        *(_QWORD *)a1 = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 8) = 1;
        *(_DWORD *)(a1 + 24) = 1;
        goto LABEL_46;
      }
      v16 = v47;
    }
    v39 = v16;
    if ( v16 > 0x40 )
      sub_C43780((__int64)&v38, &v46);
    else
      v38 = v46;
    if ( !(unsigned __int8)sub_D5D7B0(a2, &v38) )
      goto LABEL_18;
    sub_C49BE0((__int64)&v44, (__int64)&v36, (__int64)&v38, &v35);
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
    v36 = v44;
    v37 = v45;
    if ( v35 )
    {
LABEL_18:
      *(_QWORD *)a1 = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 8) = 1;
      *(_DWORD *)(a1 + 24) = 1;
      goto LABEL_19;
    }
    v25 = *(_WORD *)(a3 + 2);
    v41 = v45;
    _BitScanReverse64(&v26, 1LL << v25);
    LOBYTE(v27) = 63 - (v26 ^ 0x3F);
    HIBYTE(v27) = 1;
    if ( v45 > 0x40 )
    {
      v34 = v27;
      sub_C43780((__int64)&v40, &v36);
      v27 = v34;
    }
    else
    {
      v40 = v44;
    }
    sub_D5D630((__int64)&v42, a2, &v40, v27);
    v28 = *(_DWORD *)(a2 + 48);
    v45 = v28;
    if ( v28 > 0x40 )
    {
      sub_C43780((__int64)&v44, (const void **)(a2 + 40));
      v31 = v45;
      *(_DWORD *)(a1 + 8) = v45;
      if ( v31 > 0x40 )
      {
        sub_C43780(a1, &v44);
        goto LABEL_65;
      }
    }
    else
    {
      v29 = *(const void **)(a2 + 40);
      *(_DWORD *)(a1 + 8) = v28;
      v44 = v29;
    }
    *(_QWORD *)a1 = v44;
LABEL_65:
    v30 = v43;
    *(_DWORD *)(a1 + 24) = v43;
    if ( v30 > 0x40 )
      sub_C43780(a1 + 16, &v42);
    else
      *(_QWORD *)(a1 + 16) = v42;
    if ( v45 > 0x40 && v44 )
      j_j___libc_free_0_0(v44);
    if ( v43 > 0x40 && v42 )
      j_j___libc_free_0_0(v42);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
LABEL_19:
    if ( v39 > 0x40 && v38 )
      j_j___libc_free_0_0(v38);
    if ( !v48 )
      goto LABEL_46;
    v48 = 0;
    if ( (unsigned int)v47 <= 0x40 )
      goto LABEL_46;
    v17 = v46;
    if ( !v46 )
      goto LABEL_46;
    goto LABEL_25;
  }
LABEL_28:
  _BitScanReverse64(&v18, 1LL << *(_WORD *)(a3 + 2));
  LOBYTE(v19) = 63 - (v18 ^ 0x3F);
  v43 = v37;
  HIBYTE(v19) = 1;
  if ( v37 > 0x40 )
  {
    v33 = v19;
    sub_C43780((__int64)&v42, &v36);
    v19 = v33;
  }
  else
  {
    v42 = v36;
  }
  sub_D5D630((__int64)&v44, a2, &v42, v19);
  v20 = *(_DWORD *)(a2 + 48);
  LODWORD(v47) = v20;
  if ( v20 <= 0x40 )
  {
    v21 = *(const void **)(a2 + 40);
    *(_DWORD *)(a1 + 8) = v20;
    v46 = v21;
LABEL_32:
    *(_QWORD *)a1 = v46;
    goto LABEL_33;
  }
  sub_C43780((__int64)&v46, (const void **)(a2 + 40));
  v24 = v47;
  *(_DWORD *)(a1 + 8) = v47;
  if ( v24 <= 0x40 )
    goto LABEL_32;
  sub_C43780(a1, &v46);
LABEL_33:
  v22 = v45;
  *(_DWORD *)(a1 + 24) = v45;
  if ( v22 > 0x40 )
    sub_C43780(a1 + 16, &v44);
  else
    *(_QWORD *)(a1 + 16) = v44;
  if ( (unsigned int)v47 > 0x40 && v46 )
    j_j___libc_free_0_0(v46);
  if ( v45 > 0x40 && v44 )
    j_j___libc_free_0_0(v44);
  if ( v43 <= 0x40 )
    goto LABEL_46;
  v17 = v42;
  if ( !v42 )
    goto LABEL_46;
LABEL_25:
  j_j___libc_free_0_0(v17);
LABEL_46:
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  return a1;
}
