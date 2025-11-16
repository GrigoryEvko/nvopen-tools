// Function: sub_145D730
// Address: 0x145d730
//
__int64 __fastcall sub_145D730(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // r14
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned int v14; // ecx
  unsigned int v15; // eax
  unsigned int v16; // ecx
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  unsigned int v19; // eax
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-118h]
  __int64 v23; // [rsp+10h] [rbp-110h]
  unsigned int v24; // [rsp+20h] [rbp-100h]
  __int64 v25; // [rsp+28h] [rbp-F8h]
  __int64 v27[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v28; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v29; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v30; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v31; // [rsp+78h] [rbp-A8h]
  unsigned __int64 v32; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v33; // [rsp+88h] [rbp-98h]
  __int64 v34[2]; // [rsp+90h] [rbp-90h] BYREF
  unsigned __int64 v35; // [rsp+A0h] [rbp-80h] BYREF
  unsigned int v36; // [rsp+A8h] [rbp-78h]
  __int64 v37; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+B8h] [rbp-68h]
  unsigned __int64 v39; // [rsp+C0h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+C8h] [rbp-58h]
  unsigned __int64 v41; // [rsp+D0h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+D8h] [rbp-48h]
  unsigned __int64 v43; // [rsp+E0h] [rbp-40h] BYREF
  unsigned int v44; // [rsp+E8h] [rbp-38h]

  v3 = *(_QWORD **)(a2 + 32);
  v4 = *v3;
  if ( *(_WORD *)(*v3 + 24LL) )
    v4 = 0;
  v5 = v3[1];
  if ( *(_WORD *)(v5 + 24) || (v6 = v3[2], *(_WORD *)(v6 + 24)) || !v4 )
  {
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  v7 = *(_QWORD *)(v4 + 32);
  v8 = v7 + 24;
  v9 = *(_QWORD *)(v6 + 32) + 24LL;
  v10 = *(_QWORD *)(v5 + 32) + 24LL;
  sub_135E0D0((__int64)v27, *(_DWORD *)(v7 + 32), 2, 0);
  sub_16A9F90(&v28, v9, v27);
  sub_13A38D0((__int64)&v30, v10);
  sub_16A7590(&v30, &v28);
  v33 = v31;
  if ( v31 > 0x40 )
    sub_16A4FD0(&v32, &v30);
  else
    v32 = v30;
  sub_16A7C10(&v32, &v30);
  sub_16A7B50(&v41, &v28, v8);
  sub_16A7A10(&v41, 4);
  v11 = v42;
  v42 = 0;
  v44 = v11;
  v43 = v41;
  sub_16A7590(&v32, &v43);
  sub_135E100((__int64 *)&v43);
  sub_135E100((__int64 *)&v41);
  if ( v33 > 0x40 )
    v12 = *(_QWORD *)(v32 + 8LL * ((v33 - 1) >> 6));
  else
    v12 = v32;
  if ( (v12 & (1LL << ((unsigned __int8)v33 - 1))) == 0 )
  {
    sub_16AA6E0(v34, &v32);
    v14 = v31;
    v31 = 0;
    v44 = v14;
    v43 = v30;
    if ( v14 <= 0x40 )
      v43 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & ~v30;
    else
      sub_16A8F40(&v43);
    sub_16A7400(&v43);
    v15 = v44;
    v44 = 0;
    v36 = v15;
    v35 = v43;
    sub_135E100((__int64 *)&v43);
    v16 = v29;
    v29 = 0;
    v38 = v16;
    v37 = v28;
    if ( v16 > 0x40 )
    {
      sub_16A7DC0(&v37, 1);
      if ( v38 > 0x40 )
      {
        v24 = v38;
        if ( v24 != (unsigned int)sub_16A57B0(&v37) )
          goto LABEL_23;
        goto LABEL_28;
      }
      v17 = v37;
    }
    else
    {
      if ( v16 == 1 )
      {
        v37 = 0;
        goto LABEL_28;
      }
      v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & (2 * v28);
      v37 = v17;
    }
    if ( v17 )
    {
LABEL_23:
      v23 = sub_15E0530(*(_QWORD *)(a3 + 24));
      sub_13A38D0((__int64)&v39, (__int64)&v35);
      sub_16A7200(&v39, v34);
      v18 = v40;
      v40 = 0;
      v42 = v18;
      v41 = v39;
      sub_16A9F90(&v43, &v41, &v37);
      v22 = sub_159C0E0(v23, &v43);
      sub_135E100((__int64 *)&v43);
      sub_135E100((__int64 *)&v41);
      sub_135E100((__int64 *)&v39);
      sub_13A38D0((__int64)&v39, (__int64)&v35);
      sub_16A7590(&v39, v34);
      v19 = v40;
      v40 = 0;
      v42 = v19;
      v41 = v39;
      sub_16A9F90(&v43, &v41, &v37);
      v25 = sub_159C0E0(v23, &v43);
      sub_135E100((__int64 *)&v43);
      sub_135E100((__int64 *)&v41);
      sub_135E100((__int64 *)&v39);
      v20 = sub_145CE20(a3, v25);
      v21 = sub_145CE20(a3, v22);
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v21;
      *(_QWORD *)(a1 + 8) = v20;
LABEL_24:
      sub_135E100(&v37);
      sub_135E100((__int64 *)&v35);
      sub_135E100(v34);
      goto LABEL_12;
    }
LABEL_28:
    *(_BYTE *)(a1 + 16) = 0;
    goto LABEL_24;
  }
  *(_BYTE *)(a1 + 16) = 0;
LABEL_12:
  sub_135E100((__int64 *)&v32);
  sub_135E100((__int64 *)&v30);
  sub_135E100(&v28);
  sub_135E100(v27);
  return a1;
}
