// Function: sub_1EF4660
// Address: 0x1ef4660
//
__int64 __fastcall sub_1EF4660(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6, __m128i a7)
{
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned int v15; // ebx
  __int64 *v16; // r12
  unsigned int v17; // r12d
  __int64 v20; // [rsp+10h] [rbp-100h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v22; // [rsp+20h] [rbp-F0h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-E8h]
  __int64 v24; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v25; // [rsp+38h] [rbp-D8h]
  __int64 v26; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v27; // [rsp+48h] [rbp-C8h]
  __int64 v28; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v29; // [rsp+58h] [rbp-B8h]
  __int64 v30; // [rsp+60h] [rbp-B0h]
  unsigned int v31; // [rsp+68h] [rbp-A8h]
  __int64 v32; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v33; // [rsp+78h] [rbp-98h]
  __int64 v34; // [rsp+80h] [rbp-90h]
  unsigned int v35; // [rsp+88h] [rbp-88h]
  unsigned __int64 v36; // [rsp+90h] [rbp-80h] BYREF
  unsigned int v37; // [rsp+98h] [rbp-78h]
  __int64 v38; // [rsp+A0h] [rbp-70h]
  unsigned int v39; // [rsp+A8h] [rbp-68h]
  _QWORD *v40[2]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v41; // [rsp+C0h] [rbp-50h]
  __int64 v42; // [rsp+C8h] [rbp-48h]
  int v43; // [rsp+D0h] [rbp-40h]
  __int64 v44; // [rsp+D8h] [rbp-38h]

  v9 = *(_QWORD **)(a1 + 24);
  v44 = a4;
  v40[0] = v9;
  v40[1] = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v10 = sub_146F1B0((__int64)v9, a2);
  v11 = sub_1EF3D60(v40, v10, a6, a7);
  v12 = *(_QWORD *)(a1 + 24);
  v13 = v11;
  v14 = sub_1456040(v11);
  v15 = sub_1456C90(v12, v14);
  v16 = sub_1477920(*(_QWORD *)(a1 + 24), v13, 0);
  v25 = *((_DWORD *)v16 + 2);
  if ( v25 > 0x40 )
    sub_16A4FD0((__int64)&v24, (const void **)v16);
  else
    v24 = *v16;
  v27 = *((_DWORD *)v16 + 6);
  if ( v27 > 0x40 )
    sub_16A4FD0((__int64)&v26, (const void **)v16 + 2);
  else
    v26 = v16[2];
  v37 = v15;
  if ( v15 > 0x40 )
  {
    sub_16A4EF0((__int64)&v36, a3, 0);
    v33 = v15;
    sub_16A4EF0((__int64)&v32, 0, 0);
  }
  else
  {
    v33 = v15;
    v32 = 0;
    v36 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & a3;
  }
  sub_15898E0((__int64)&v28, (__int64)&v32, (__int64 *)&v36);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  sub_158E130((__int64)&v32, (__int64)&v24, (__int64)&v28);
  v23 = v15;
  if ( v15 > 0x40 )
  {
    sub_16A4EF0((__int64)&v22, a5, 0);
    v21 = v15;
    sub_16A4EF0((__int64)&v20, 0, 0);
  }
  else
  {
    v21 = v15;
    v20 = 0;
    v22 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & a5;
  }
  sub_15898E0((__int64)&v36, (__int64)&v20, (__int64 *)&v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  v17 = sub_158BB40((__int64)&v36, (__int64)&v32);
  if ( v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  j___libc_free_0(v41);
  return v17;
}
