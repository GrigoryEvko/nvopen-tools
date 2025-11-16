// Function: sub_AADC60
// Address: 0xaadc60
//
__int64 __fastcall sub_AADC60(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  unsigned int v5; // r13d
  unsigned int v7; // r15d
  unsigned int v9; // r8d
  unsigned __int64 v10; // rsi
  unsigned int v11; // eax
  unsigned __int64 v12; // rax
  unsigned int v13; // ecx
  __int64 v14; // rsi
  unsigned __int64 v15; // rsi
  unsigned int v16; // ecx
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned int v19; // eax
  unsigned int v20; // ecx
  unsigned int v21; // ecx
  unsigned __int64 v22; // rax
  unsigned int v23; // ebx
  unsigned __int64 v24; // rax
  __int64 *v25; // rsi
  __int64 v26; // rdx
  unsigned int v27; // [rsp+8h] [rbp-A8h]
  unsigned int v28; // [rsp+Ch] [rbp-A4h]
  unsigned int v29; // [rsp+Ch] [rbp-A4h]
  unsigned int v32; // [rsp+18h] [rbp-98h]
  unsigned int v33; // [rsp+18h] [rbp-98h]
  char v34; // [rsp+2Fh] [rbp-81h] BYREF
  __int64 v35; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-78h]
  __int64 v37; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+48h] [rbp-68h]
  __int64 v39; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+58h] [rbp-58h]
  __int64 v41; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-48h]
  __int64 v43; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v44; // [rsp+78h] [rbp-38h]

  v5 = a4;
  v7 = *(_DWORD *)(a2 + 8);
  sub_C47E60(&v35, a2, a4, &v34);
  if ( v34 )
  {
    sub_AADB10(a1, v7, 0);
    goto LABEL_3;
  }
  v38 = v36;
  if ( v36 > 0x40 )
    sub_C43780(&v37, &v35);
  else
    v37 = v35;
  v9 = *(_DWORD *)(a3 + 8);
  if ( v9 <= 0x40 )
  {
    v10 = *(_QWORD *)a3;
    v11 = *(_DWORD *)(a3 + 8);
    if ( *(_QWORD *)a3 )
    {
      _BitScanReverse64(&v12, v10);
      v11 = v9 - 64 + (v12 ^ 0x3F);
    }
    v13 = v11 - 1;
    if ( v5 <= v11 - 1 )
    {
      v44 = *(_DWORD *)(a3 + 8);
      v43 = v10;
      if ( a5 <= v13 )
        v13 = a5;
      goto LABEL_15;
    }
LABEL_47:
    if ( v5 < v11 )
      v5 = v11;
    v17 = *(_DWORD *)(a2 + 8);
    if ( v17 > 0x40 )
      goto LABEL_26;
LABEL_50:
    v21 = v17 - 64;
    if ( *(_QWORD *)a2 )
    {
      _BitScanReverse64(&v22, *(_QWORD *)a2);
      v17 = v21 + (v22 ^ 0x3F);
    }
    goto LABEL_27;
  }
  v28 = *(_DWORD *)(a3 + 8);
  v11 = sub_C444A0(a3);
  v20 = v11 - 1;
  if ( v5 > v11 - 1 )
    goto LABEL_47;
  v27 = v11;
  v44 = v28;
  if ( a5 <= v20 )
    v20 = a5;
  v29 = v20;
  sub_C43780(&v43, a3);
  v9 = v44;
  v13 = v29;
  v11 = v27;
  if ( v44 > 0x40 )
  {
    sub_C47690(&v43, v29);
    v11 = v27;
    goto LABEL_20;
  }
LABEL_15:
  v14 = 0;
  if ( v13 != v9 )
    v14 = v43 << v13;
  v15 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v9) & v14;
  if ( !v9 )
    v15 = 0;
  v43 = v15;
LABEL_20:
  if ( v38 > 0x40 && v37 )
  {
    v32 = v11;
    j_j___libc_free_0_0(v37);
    v11 = v32;
  }
  v33 = v11;
  v37 = v43;
  v16 = v44;
  v44 = 0;
  v38 = v16;
  sub_969240(&v43);
  if ( v5 < v33 )
    v5 = v33;
  v17 = *(_DWORD *)(a2 + 8);
  if ( v17 <= 0x40 )
    goto LABEL_50;
LABEL_26:
  v17 = sub_C444A0(a2);
LABEL_27:
  v18 = v17 - 1;
  if ( v18 > a5 )
    v18 = a5;
  if ( v18 >= v5 )
  {
    v23 = v7 - 1;
    sub_9691E0((__int64)&v43, v7, 0, 0, 0);
    if ( v7 - 1 != v5 )
    {
      if ( v5 > 0x3F || v23 > 0x40 )
      {
        sub_C43C90(&v43, v5, v23);
      }
      else
      {
        v24 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v5 - (unsigned __int8)v23 + 64) << v5;
        if ( v44 > 0x40 )
          *(_QWORD *)v43 |= v24;
        else
          v43 |= v24;
      }
    }
    v25 = &v43;
    if ( (int)sub_C49970(&v37, &v43) > 0 )
      v25 = &v37;
    if ( v38 <= 0x40 && *((_DWORD *)v25 + 2) <= 0x40u )
    {
      v26 = *v25;
      v38 = *((_DWORD *)v25 + 2);
      v37 = v26;
    }
    else
    {
      sub_C43990(&v37, v25);
    }
    sub_969240(&v43);
  }
  v40 = v38;
  if ( v38 > 0x40 )
    sub_C43780(&v39, &v37);
  else
    v39 = v37;
  sub_C46A40(&v39, 1);
  v19 = v40;
  v40 = 0;
  v42 = v19;
  v41 = v39;
  v44 = v36;
  if ( v36 > 0x40 )
    sub_C43780(&v43, &v35);
  else
    v43 = v35;
  sub_9875E0(a1, &v43, &v41);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
LABEL_3:
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  return a1;
}
