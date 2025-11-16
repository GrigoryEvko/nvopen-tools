// Function: sub_C6F890
// Address: 0xc6f890
//
__int64 __fastcall sub_C6F890(__int64 a1, unsigned __int64 *a2, unsigned __int64 *a3, unsigned __int8 a4)
{
  unsigned int v6; // r13d
  unsigned int v7; // r15d
  unsigned int v8; // edx
  unsigned __int64 v9; // r8
  unsigned int v10; // ecx
  unsigned __int64 v11; // rsi
  bool v12; // cc
  unsigned __int64 v13; // rdi
  unsigned int v14; // r15d
  unsigned __int64 v15; // rcx
  unsigned int v16; // ebx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rdi
  unsigned int v19; // edx
  unsigned __int64 v20; // rdi
  unsigned int v21; // r12d
  unsigned __int64 v22; // rbx
  unsigned int v23; // r13d
  unsigned __int64 v24; // r15
  unsigned __int64 v25; // rdi
  int v27; // edx
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v30; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v31; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v32; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v33; // [rsp+8h] [rbp-A8h]
  unsigned int v34; // [rsp+10h] [rbp-A0h]
  unsigned int v35; // [rsp+10h] [rbp-A0h]
  unsigned int v36; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v37; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v38; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v39; // [rsp+10h] [rbp-A0h]
  unsigned int v40; // [rsp+10h] [rbp-A0h]
  unsigned int v41; // [rsp+18h] [rbp-98h]
  unsigned int v42; // [rsp+18h] [rbp-98h]
  unsigned __int64 v45; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v46; // [rsp+48h] [rbp-68h]
  unsigned __int64 v47; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v48; // [rsp+58h] [rbp-58h]
  unsigned __int64 v49; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v50; // [rsp+68h] [rbp-48h]
  unsigned __int64 v51; // [rsp+70h] [rbp-40h]
  int v52; // [rsp+78h] [rbp-38h]

  v6 = *((_DWORD *)a2 + 2);
  v7 = v6 + 1;
  sub_C449B0((__int64)&v45, (const void **)a2, v6 + 1);
  if ( v6 != v46 )
  {
    if ( v6 > 0x3F || v46 > 0x40 )
      sub_C43C90(&v45, v6, v46);
    else
      v45 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v6 - (unsigned __int8)v46 + 64) << v6;
  }
  sub_C449B0((__int64)&v49, (const void **)a2 + 2, v7);
  v8 = v46;
  v48 = v46;
  if ( v46 > 0x40 )
  {
    sub_C43780((__int64)&v47, (const void **)&v45);
    v8 = v48;
    v9 = v47;
    v10 = v50;
    v11 = v49;
    if ( v46 > 0x40 && v45 )
    {
      v42 = v48;
      v30 = v47;
      v33 = v49;
      v40 = v50;
      j_j___libc_free_0_0(v45);
      v8 = v42;
      v9 = v30;
      v11 = v33;
      v10 = v40;
    }
  }
  else
  {
    v9 = v45;
    v10 = v50;
    v11 = v49;
  }
  if ( *((_DWORD *)a2 + 2) > 0x40u && *a2 )
  {
    v41 = v8;
    v29 = v9;
    v34 = v10;
    j_j___libc_free_0_0(*a2);
    v8 = v41;
    v9 = v29;
    v10 = v34;
  }
  v12 = *((_DWORD *)a2 + 6) <= 0x40u;
  *a2 = v9;
  *((_DWORD *)a2 + 2) = v8;
  if ( !v12 )
  {
    v13 = a2[2];
    if ( v13 )
    {
      v35 = v10;
      j_j___libc_free_0_0(v13);
      v10 = v35;
    }
  }
  a2[2] = v11;
  *((_DWORD *)a2 + 6) = v10;
  v36 = *((_DWORD *)a3 + 2);
  sub_C449B0((__int64)&v45, (const void **)a3, v7);
  if ( v36 != v46 )
  {
    if ( v36 > 0x3F || v46 > 0x40 )
      sub_C43C90(&v45, v36, v46);
    else
      v45 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v36 - (unsigned __int8)v46 + 64) << v36;
  }
  sub_C449B0((__int64)&v49, (const void **)a3 + 2, v7);
  v14 = v46;
  v48 = v46;
  if ( v46 > 0x40 )
  {
    sub_C43780((__int64)&v47, (const void **)&v45);
    v14 = v48;
    v15 = v47;
    v16 = v50;
    v17 = v49;
    if ( v46 > 0x40 && v45 )
    {
      v32 = v47;
      v39 = v49;
      j_j___libc_free_0_0(v45);
      v15 = v32;
      v17 = v39;
    }
  }
  else
  {
    v15 = v45;
    v16 = v50;
    v17 = v49;
  }
  if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
  {
    v31 = v15;
    v37 = v17;
    j_j___libc_free_0_0(*a3);
    v15 = v31;
    v17 = v37;
  }
  v12 = *((_DWORD *)a3 + 6) <= 0x40u;
  *a3 = v15;
  *((_DWORD *)a3 + 2) = v14;
  if ( !v12 )
  {
    v18 = a3[2];
    if ( v18 )
    {
      v38 = v17;
      j_j___libc_free_0_0(v18);
      v17 = v38;
    }
  }
  a3[2] = v17;
  *((_DWORD *)a3 + 6) = v16;
  sub_C6EF30((__int64)&v49, (__int64)a2, (__int64)a3, a4 ^ 1, a4);
  if ( *((_DWORD *)a2 + 2) > 0x40u && *a2 )
    j_j___libc_free_0_0(*a2);
  v12 = *((_DWORD *)a2 + 6) <= 0x40u;
  *a2 = v49;
  v19 = v50;
  v50 = 0;
  *((_DWORD *)a2 + 2) = v19;
  if ( v12 || (v20 = a2[2]) == 0 )
  {
    a2[2] = v51;
    *((_DWORD *)a2 + 6) = v52;
  }
  else
  {
    j_j___libc_free_0_0(v20);
    v12 = v50 <= 0x40;
    a2[2] = v51;
    *((_DWORD *)a2 + 6) = v52;
    if ( !v12 && v49 )
      j_j___libc_free_0_0(v49);
  }
  sub_C440A0((__int64)&v49, (__int64 *)a2 + 2, v6, 1u);
  sub_C440A0((__int64)&v47, (__int64 *)a2, v6, 1u);
  v21 = v48;
  v22 = v47;
  v23 = v50;
  v24 = v49;
  if ( *((_DWORD *)a2 + 2) > 0x40u && *a2 )
    j_j___libc_free_0_0(*a2);
  v12 = *((_DWORD *)a2 + 6) <= 0x40u;
  *a2 = v22;
  *((_DWORD *)a2 + 2) = v21;
  if ( !v12 )
  {
    v25 = a2[2];
    if ( v25 )
    {
      j_j___libc_free_0_0(v25);
      v21 = *((_DWORD *)a2 + 2);
      v22 = *a2;
    }
  }
  *((_DWORD *)a2 + 6) = v23;
  a2[2] = v24;
  *(_DWORD *)(a1 + 8) = v21;
  *(_QWORD *)a1 = v22;
  v27 = *((_DWORD *)a2 + 6);
  *((_DWORD *)a2 + 2) = 0;
  *(_DWORD *)(a1 + 24) = v27;
  v28 = a2[2];
  *((_DWORD *)a2 + 6) = 0;
  *(_QWORD *)(a1 + 16) = v28;
  return a1;
}
