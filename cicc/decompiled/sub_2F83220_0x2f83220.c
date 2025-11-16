// Function: sub_2F83220
// Address: 0x2f83220
//
__int64 __fastcall sub_2F83220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v8; // r15
  __int64 v9; // rax
  unsigned int v10; // r8d
  _QWORD *v12; // rax
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned int v16; // r12d
  __int64 v17; // rbx
  unsigned int v18; // eax
  unsigned __int8 v19; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v20; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v21; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v22; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v23; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v24; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v25; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v26; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v27; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v28; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v29; // [rsp+20h] [rbp-C0h] BYREF
  unsigned int v30; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v31; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v33; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v34; // [rsp+48h] [rbp-98h]
  unsigned __int64 v35; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v36; // [rsp+58h] [rbp-88h]
  unsigned __int64 v37; // [rsp+60h] [rbp-80h]
  unsigned int v38; // [rsp+68h] [rbp-78h]
  unsigned __int64 v39; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v40; // [rsp+78h] [rbp-68h]
  unsigned __int64 v41; // [rsp+80h] [rbp-60h]
  unsigned int v42; // [rsp+88h] [rbp-58h]
  unsigned __int64 v43; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v44; // [rsp+98h] [rbp-48h]
  unsigned __int64 v45; // [rsp+A0h] [rbp-40h]
  unsigned int v46; // [rsp+A8h] [rbp-38h]

  v8 = sub_DD8400(*(_QWORD *)(a1 + 32), a2);
  v9 = sub_D97190(*(_QWORD *)(a1 + 32), (__int64)v8);
  v10 = 0;
  if ( *(_WORD *)(v9 + 24) == 15 && a4 == *(_QWORD *)(v9 - 8) )
  {
    v12 = sub_DCB010(*(__int64 **)(a1 + 32), (__int64)v8);
    v13 = *(_QWORD *)(a1 + 32);
    v14 = (__int64)v12;
    v15 = sub_D95540((__int64)v12);
    v16 = sub_D97050(v13, v15);
    v17 = sub_DBB9F0(*(_QWORD *)(a1 + 32), v14, 0, 0);
    v32 = *(_DWORD *)(v17 + 8);
    if ( v32 > 0x40 )
      sub_C43780((__int64)&v31, (const void **)v17);
    else
      v31 = *(_QWORD *)v17;
    v34 = *(_DWORD *)(v17 + 24);
    if ( v34 > 0x40 )
      sub_C43780((__int64)&v33, (const void **)(v17 + 16));
    else
      v33 = *(_QWORD *)(v17 + 16);
    v44 = v16;
    if ( v16 > 0x40 )
    {
      sub_C43690((__int64)&v43, a3, 0);
      v40 = v16;
      sub_C43690((__int64)&v39, 0, 0);
    }
    else
    {
      v40 = v16;
      v39 = 0;
      v43 = a3;
    }
    sub_AADC30((__int64)&v35, (__int64)&v39, (__int64 *)&v43);
    if ( v40 > 0x40 && v39 )
      j_j___libc_free_0_0(v39);
    if ( v44 > 0x40 && v43 )
      j_j___libc_free_0_0(v43);
    sub_AB4F10((__int64)&v39, (__int64)&v31, (__int64)&v35);
    v30 = v16;
    if ( v16 > 0x40 )
    {
      sub_C43690((__int64)&v29, a5, 0);
      v28 = v16;
      sub_C43690((__int64)&v27, 0, 0);
    }
    else
    {
      v29 = a5;
      v27 = 0;
      v28 = v16;
    }
    sub_AADC30((__int64)&v43, (__int64)&v27, (__int64 *)&v29);
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    v18 = sub_AB1BB0((__int64)&v43, (__int64)&v39);
    v10 = v18;
    if ( v46 > 0x40 && v45 )
    {
      v19 = v18;
      j_j___libc_free_0_0(v45);
      v10 = v19;
    }
    if ( v44 > 0x40 && v43 )
    {
      v20 = v10;
      j_j___libc_free_0_0(v43);
      v10 = v20;
    }
    if ( v42 > 0x40 && v41 )
    {
      v21 = v10;
      j_j___libc_free_0_0(v41);
      v10 = v21;
    }
    if ( v40 > 0x40 && v39 )
    {
      v22 = v10;
      j_j___libc_free_0_0(v39);
      v10 = v22;
    }
    if ( v38 > 0x40 && v37 )
    {
      v23 = v10;
      j_j___libc_free_0_0(v37);
      v10 = v23;
    }
    if ( v36 > 0x40 && v35 )
    {
      v24 = v10;
      j_j___libc_free_0_0(v35);
      v10 = v24;
    }
    if ( v34 > 0x40 && v33 )
    {
      v25 = v10;
      j_j___libc_free_0_0(v33);
      v10 = v25;
    }
    if ( v32 > 0x40 && v31 )
    {
      v26 = v10;
      j_j___libc_free_0_0(v31);
      return v26;
    }
  }
  return v10;
}
