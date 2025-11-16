// Function: sub_AAE090
// Address: 0xaae090
//
__int64 __fastcall sub_AAE090(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  __int64 v8; // r9
  unsigned int v10; // edx
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  unsigned int v13; // ecx
  unsigned int v14; // eax
  unsigned int v15; // ecx
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  unsigned int v19; // edx
  unsigned int v20; // eax
  unsigned int v21; // ecx
  unsigned int v22; // eax
  unsigned __int64 v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // eax
  __int64 v26; // [rsp+8h] [rbp-A8h]
  unsigned int v27; // [rsp+8h] [rbp-A8h]
  unsigned int v28; // [rsp+10h] [rbp-A0h]
  unsigned int v29; // [rsp+14h] [rbp-9Ch]
  unsigned int v30; // [rsp+18h] [rbp-98h]
  char v31; // [rsp+2Fh] [rbp-81h] BYREF
  __int64 v32; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-78h]
  __int64 v34; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-68h]
  __int64 v36; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v37; // [rsp+58h] [rbp-58h]
  __int64 v38; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v39; // [rsp+68h] [rbp-48h]
  __int64 v40; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v41; // [rsp+78h] [rbp-38h]

  v29 = *(_DWORD *)(a2 + 8);
  sub_C47E60(&v32, a3, a4, &v31);
  v8 = a2;
  if ( v31 )
  {
    sub_AADB10(a1, v29, 0);
    goto LABEL_3;
  }
  v35 = v33;
  if ( v33 > 0x40 )
  {
    sub_C43780(&v34, &v32);
    v8 = a2;
  }
  else
  {
    v34 = v32;
  }
  v10 = *(_DWORD *)(v8 + 8);
  if ( v10 > 0x40 )
  {
    v28 = *(_DWORD *)(v8 + 8);
    v26 = v8;
    v21 = sub_C44500(v8);
    v22 = v21 - 1;
    if ( a4 >= v21 )
      v21 = a4;
    v30 = v21;
    if ( a4 <= v22 )
    {
      v24 = v26;
      v41 = v28;
      if ( v22 > a5 )
        v22 = a5;
      v27 = v22;
      sub_C43780(&v40, v24);
      v10 = v41;
      v15 = v27;
      if ( v41 > 0x40 )
      {
        sub_C47690(&v40, v27);
        goto LABEL_25;
      }
LABEL_20:
      v16 = 0;
      if ( v15 != v10 )
        v16 = v40 << v15;
      v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v10) & v16;
      if ( !v10 )
        v17 = 0;
      v40 = v17;
LABEL_25:
      if ( v35 > 0x40 && v34 )
        j_j___libc_free_0_0(v34);
      v34 = v40;
      v35 = v41;
      v18 = *(_DWORD *)(a3 + 8);
      if ( v18 > 0x40 )
        goto LABEL_29;
      goto LABEL_53;
    }
  }
  else
  {
    v11 = *(_QWORD *)v8;
    if ( !v10 )
    {
      v41 = 0;
      v15 = a5;
      v30 = a4;
LABEL_19:
      v40 = v11;
      goto LABEL_20;
    }
    if ( v11 << (64 - (unsigned __int8)v10) == -1 )
    {
      v14 = 63;
      v13 = 64;
    }
    else
    {
      _BitScanReverse64(&v12, ~(v11 << (64 - (unsigned __int8)v10)));
      v13 = v12 ^ 0x3F;
      v14 = (v12 ^ 0x3F) - 1;
    }
    if ( a4 >= v13 )
      v13 = a4;
    v30 = v13;
    if ( v14 >= a4 )
    {
      v41 = *(_DWORD *)(v8 + 8);
      if ( v14 > a5 )
        v14 = a5;
      v15 = v14;
      goto LABEL_19;
    }
  }
  v18 = *(_DWORD *)(a3 + 8);
  if ( v18 > 0x40 )
  {
LABEL_29:
    v19 = sub_C44500(a3) - 1;
LABEL_30:
    if ( a5 > v19 )
      a5 = v19;
    goto LABEL_32;
  }
LABEL_53:
  if ( v18 )
  {
    v19 = 63;
    v23 = ~(*(_QWORD *)a3 << (64 - (unsigned __int8)v18));
    if ( v23 )
    {
      _BitScanReverse64(&v23, v23);
      v19 = (v23 ^ 0x3F) - 1;
    }
    goto LABEL_30;
  }
LABEL_32:
  if ( a5 >= v30 )
  {
    sub_986680((__int64)&v40, v29);
    if ( v35 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    v34 = v40;
    v25 = v41;
    v41 = 0;
    v35 = v25;
    sub_969240(&v40);
  }
  v37 = v33;
  if ( v33 > 0x40 )
    sub_C43780(&v36, &v32);
  else
    v36 = v32;
  sub_C46A40(&v36, 1);
  v20 = v37;
  v37 = 0;
  v39 = v20;
  v38 = v36;
  v41 = v35;
  if ( v35 > 0x40 )
    sub_C43780(&v40, &v34);
  else
    v40 = v34;
  sub_9875E0(a1, &v40, &v38);
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
LABEL_3:
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  return a1;
}
