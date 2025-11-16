// Function: sub_D94A80
// Address: 0xd94a80
//
__int64 __fastcall sub_D94A80(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, bool a5)
{
  bool v7; // dl
  unsigned int v8; // r14d
  unsigned int v9; // r14d
  unsigned int v10; // eax
  unsigned int v11; // eax
  int v13; // eax
  int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned __int64 v17; // rdx
  int v18; // eax
  unsigned int v19; // eax
  unsigned int v20; // r13d
  __int64 v21; // r12
  __int64 *v22; // rax
  unsigned int v23; // edx
  unsigned int v24; // eax
  unsigned __int64 v25; // rax
  unsigned int v26; // eax
  int v27; // [rsp+8h] [rbp-B8h]
  int v28; // [rsp+8h] [rbp-B8h]
  bool v30; // [rsp+10h] [rbp-B0h]
  bool v31; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v32; // [rsp+10h] [rbp-B0h]
  bool v33; // [rsp+18h] [rbp-A8h]
  unsigned int v34; // [rsp+18h] [rbp-A8h]
  unsigned int v35; // [rsp+1Ch] [rbp-A4h]
  unsigned __int64 v36; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v37; // [rsp+28h] [rbp-98h]
  __int64 v38; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v39; // [rsp+38h] [rbp-88h]
  __int64 v40; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v41; // [rsp+48h] [rbp-78h]
  __int64 v42; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v43; // [rsp+58h] [rbp-68h]
  __int64 v44; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v45; // [rsp+68h] [rbp-58h]
  unsigned __int64 v46; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v47; // [rsp+78h] [rbp-48h]
  __int64 v48; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v49; // [rsp+88h] [rbp-38h]

  v7 = a5;
  v8 = *((_DWORD *)a2 + 2);
  v33 = a5;
  v35 = v8;
  if ( v8 > 0x40 )
  {
    v13 = sub_C444A0((__int64)a2);
    v7 = a5;
    if ( v8 - v13 <= 0x40 && !*(_QWORD *)*a2 )
      goto LABEL_5;
  }
  else if ( !*a2 )
  {
    goto LABEL_5;
  }
  v9 = *(_DWORD *)(a4 + 8);
  if ( v9 > 0x40 )
  {
    v30 = v7;
    v14 = sub_C444A0(a4);
    v7 = v30;
    if ( v9 - v14 > 0x40 || **(_QWORD **)a4 )
      goto LABEL_15;
LABEL_5:
    v10 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a1 + 8) = v10;
    if ( v10 > 0x40 )
      sub_C43780(a1, (const void **)a3);
    else
      *(_QWORD *)a1 = *(_QWORD *)a3;
    v11 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(a1 + 24) = v11;
    if ( v11 > 0x40 )
      sub_C43780(a1 + 16, (const void **)(a3 + 16));
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
    return a1;
  }
  if ( !*(_QWORD *)a4 )
    goto LABEL_5;
LABEL_15:
  v31 = v7;
  if ( sub_AAF760(a3) )
    goto LABEL_64;
  if ( v31 )
  {
    v15 = *((_DWORD *)a2 + 2);
    v34 = v15 - 1;
    if ( v15 <= 0x40 )
      v32 = *a2;
    else
      v32 = *(_QWORD *)(*a2 + 8LL * (v34 >> 6));
    sub_9692E0((__int64)&v48, a2);
    if ( *((_DWORD *)a2 + 2) > 0x40u && *a2 )
      j_j___libc_free_0_0(*a2);
    v33 = ((1LL << v34) & v32) != 0;
    *a2 = v48;
    *((_DWORD *)a2 + 2) = v49;
  }
  v16 = *(_DWORD *)(a3 + 8);
  v47 = v16;
  if ( v16 > 0x40 )
  {
    sub_C43690((__int64)&v46, -1, 1);
  }
  else
  {
    v17 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v16;
    if ( !v16 )
      v17 = 0;
    v46 = v17;
  }
  sub_C4A1D0((__int64)&v48, (__int64)&v46, (__int64)a2);
  v18 = sub_C49970((__int64)&v48, (unsigned __int64 *)a4);
  if ( v49 > 0x40 && v48 )
  {
    v27 = v18;
    j_j___libc_free_0_0(v48);
    v18 = v27;
  }
  if ( v47 > 0x40 && v46 )
  {
    v28 = v18;
    j_j___libc_free_0_0(v46);
    v18 = v28;
  }
  if ( v18 < 0 )
  {
LABEL_64:
    sub_AADB10(a1, v35, 1);
    return a1;
  }
  sub_C472A0((__int64)&v36, (__int64)a2, (__int64 *)a4);
  v39 = *(_DWORD *)(a3 + 8);
  if ( v39 > 0x40 )
    sub_C43780((__int64)&v38, (const void **)a3);
  else
    v38 = *(_QWORD *)a3;
  v49 = *(_DWORD *)(a3 + 24);
  if ( v49 > 0x40 )
    sub_C43780((__int64)&v48, (const void **)(a3 + 16));
  else
    v48 = *(_QWORD *)(a3 + 16);
  sub_C46F20((__int64)&v48, 1u);
  v41 = v49;
  v40 = v48;
  if ( v33 )
  {
    if ( v37 > 0x40 )
    {
      sub_C43D10((__int64)&v36);
    }
    else
    {
      v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v37) & ~v36;
      if ( !v37 )
        v25 = 0;
      v36 = v25;
    }
    sub_C46250((__int64)&v36);
    sub_C45EE0((__int64)&v36, &v38);
    v26 = v37;
    v37 = 0;
    v43 = v26;
    v42 = v36;
    if ( !sub_AB1B10(a3, (__int64)&v42) )
    {
      v20 = v43;
      v21 = v42;
      v22 = &v40;
      v43 = 0;
LABEL_41:
      v23 = *((_DWORD *)v22 + 2);
      *((_DWORD *)v22 + 2) = 0;
      v45 = v23;
      v44 = *v22;
      sub_C46A40((__int64)&v44, 1);
      v24 = v45;
      v45 = 0;
      v49 = v24;
      v47 = v20;
      v48 = v44;
      v46 = v21;
      sub_9875E0(a1, (__int64 *)&v46, &v48);
      if ( v47 > 0x40 && v46 )
        j_j___libc_free_0_0(v46);
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      if ( v45 > 0x40 && v44 )
        j_j___libc_free_0_0(v44);
      goto LABEL_50;
    }
  }
  else
  {
    sub_C45EE0((__int64)&v36, &v40);
    v19 = v37;
    v37 = 0;
    v43 = v19;
    v42 = v36;
    if ( !sub_AB1B10(a3, (__int64)&v42) )
    {
      v20 = v39;
      v21 = v38;
      v22 = &v42;
      v39 = 0;
      goto LABEL_41;
    }
  }
  sub_AADB10(a1, v35, 1);
LABEL_50:
  if ( v43 > 0x40 && v42 )
    j_j___libc_free_0_0(v42);
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  return a1;
}
