// Function: sub_13AA420
// Address: 0x13aa420
//
__int64 __fastcall sub_13AA420(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rbx
  unsigned int v13; // eax
  __int64 v14; // rsi
  unsigned int v15; // r14d
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rcx
  unsigned int v23; // eax
  unsigned int v24; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // eax
  unsigned int v31; // eax
  unsigned int v32; // eax
  unsigned int v33; // eax
  __int64 *v34; // rdi
  unsigned int v35; // eax
  unsigned int v36; // eax
  char v37; // [rsp+Fh] [rbp-171h]
  char v38; // [rsp+30h] [rbp-150h]
  unsigned int v39; // [rsp+48h] [rbp-138h]
  __int64 v41; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-118h]
  __int64 v43; // [rsp+70h] [rbp-110h] BYREF
  int v44; // [rsp+78h] [rbp-108h]
  __int64 v45; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v46; // [rsp+88h] [rbp-F8h]
  __int64 v47; // [rsp+90h] [rbp-F0h] BYREF
  unsigned int v48; // [rsp+98h] [rbp-E8h]
  __int64 v49; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int v50; // [rsp+A8h] [rbp-D8h]
  unsigned __int64 v51; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned int v52; // [rsp+B8h] [rbp-C8h]
  unsigned __int64 v53; // [rsp+C0h] [rbp-C0h] BYREF
  unsigned int v54; // [rsp+C8h] [rbp-B8h]
  unsigned __int64 v55; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned int v56; // [rsp+D8h] [rbp-A8h]
  __int64 v57; // [rsp+E0h] [rbp-A0h] BYREF
  unsigned int v58; // [rsp+E8h] [rbp-98h]
  __int64 v59[2]; // [rsp+F0h] [rbp-90h] BYREF
  unsigned __int64 v60; // [rsp+100h] [rbp-80h] BYREF
  unsigned int v61; // [rsp+108h] [rbp-78h]
  unsigned __int64 v62; // [rsp+110h] [rbp-70h] BYREF
  unsigned int v63; // [rsp+118h] [rbp-68h]
  __int64 v64[2]; // [rsp+120h] [rbp-60h] BYREF
  __int64 v65[2]; // [rsp+130h] [rbp-50h] BYREF
  __int64 v66[8]; // [rsp+140h] [rbp-40h] BYREF

  LODWORD(v9) = 0;
  *(_BYTE *)(a8 + 43) = 0;
  v10 = sub_14806B0(*(_QWORD *)(a1 + 8), a5, a4, 0, 0);
  if ( *(_WORD *)(v10 + 24) || *(_WORD *)(a2 + 24) || *(_WORD *)(a3 + 24) )
    return (unsigned int)v9;
  v11 = *(_QWORD *)(a2 + 32);
  v12 = v10;
  v42 = 1;
  v41 = 0;
  v44 = 1;
  v13 = *(_DWORD *)(v11 + 32);
  v43 = 0;
  v46 = 1;
  v45 = 0;
  v48 = v13;
  if ( v13 > 0x40 )
    sub_16A4FD0(&v47, v11 + 24);
  else
    v47 = *(_QWORD *)(v11 + 24);
  v14 = *(_QWORD *)(a3 + 32);
  v50 = *(_DWORD *)(v14 + 32);
  if ( v50 > 0x40 )
    sub_16A4FD0(&v49, v14 + 24);
  else
    v49 = *(_QWORD *)(v14 + 24);
  v15 = v48;
  LODWORD(v9) = sub_13A3F30(
                  v48,
                  (__int64)&v47,
                  (__int64)&v49,
                  *(_QWORD *)(v12 + 32) + 24LL,
                  (__int64)&v41,
                  (__int64)&v43,
                  (__int64)&v45);
  if ( !(_BYTE)v9 )
  {
    v52 = v15;
    v39 = v15 - 1;
    v38 = (v15 - 1) & 0x3F;
    if ( v15 <= 0x40 )
    {
      v51 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & 1;
      v26 = sub_1456040(v12);
      v27 = sub_13A7B50(a1, a6, v26);
      if ( v27 )
      {
        sub_13A36B0((__int64)&v51, (__int64 *)(*(_QWORD *)(v27 + 32) + 24LL));
        v54 = v15;
        v37 = 1;
      }
      else
      {
        v54 = v15;
        v37 = 0;
      }
      v53 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & 1;
      v28 = sub_1456040(v12);
      v29 = sub_13A7B50(a1, a7, v28);
      if ( v29 )
      {
        LODWORD(v9) = 1;
        sub_13A36B0((__int64)&v53, (__int64 *)(*(_QWORD *)(v29 + 32) + 24LL));
      }
      v56 = v15;
      v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
      v21 = 1LL << v38;
      v22 = ~(1LL << v38);
    }
    else
    {
      sub_16A4EF0(&v51, 1, 1);
      v16 = sub_1456040(v12);
      v17 = sub_13A7B50(a1, a6, v16);
      if ( v17 )
      {
        sub_13A36B0((__int64)&v51, (__int64 *)(*(_QWORD *)(v17 + 32) + 24LL));
        v54 = v15;
        v37 = 1;
      }
      else
      {
        v54 = v15;
        v37 = 0;
      }
      sub_16A4EF0(&v53, 1, 1);
      v18 = sub_1456040(v12);
      v19 = sub_13A7B50(a1, a7, v18);
      if ( v19 )
      {
        LODWORD(v9) = 1;
        sub_13A36B0((__int64)&v53, (__int64 *)(*(_QWORD *)(v19 + 32) + 24LL));
      }
      v56 = v15;
      sub_16A4EF0(&v55, -1, 1);
      v20 = v55;
      v21 = 1LL << v38;
      v22 = ~(1LL << v38);
      if ( v56 > 0x40 )
      {
        *(_QWORD *)(v55 + 8LL * (v39 >> 6)) &= v22;
        v58 = v15;
        goto LABEL_16;
      }
    }
    v58 = v15;
    v55 = v22 & v20;
    if ( v15 <= 0x40 )
    {
      v57 = 0;
      goto LABEL_47;
    }
LABEL_16:
    sub_16A4EF0(&v57, 0, 0);
    if ( v58 > 0x40 )
    {
      *(_QWORD *)(v57 + 8LL * (v39 >> 6)) |= v21;
      goto LABEL_18;
    }
LABEL_47:
    v57 |= v21;
LABEL_18:
    sub_16A9F90(v59, &v49, &v41);
    if ( sub_13A39D0((__int64)v59, 0) )
    {
      sub_13A38D0((__int64)&v60, (__int64)&v43);
      if ( v61 > 0x40 )
        sub_16A8F40(&v60);
      else
        v60 = ~v60 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v61);
      sub_16A7400(&v60);
      v23 = v61;
      v61 = 0;
      v63 = v23;
      v62 = v60;
      sub_13A3A60((__int64)v64, (__int64)&v62, (__int64)v59);
      sub_13A38D0((__int64)v65, (__int64)&v57);
      sub_13A37A0((__int64)v66, (__int64)v65, (__int64)v64);
      sub_13A3610(&v57, v66);
      sub_135E100(v66);
      sub_135E100(v65);
      sub_135E100(v64);
      sub_135E100((__int64 *)&v62);
      sub_135E100((__int64 *)&v60);
      if ( !v37 )
        goto LABEL_22;
      sub_13A38D0((__int64)&v60, (__int64)&v51);
      sub_16A7590(&v60, &v43);
      v36 = v61;
      v61 = 0;
      v63 = v36;
      v62 = v60;
      sub_13A3C50((__int64)v64, (__int64)&v62, (__int64)v59);
      sub_13A38D0((__int64)v65, (__int64)&v55);
      sub_13A3810((__int64)v66, (__int64)v65, (__int64)v64);
      v34 = (__int64 *)&v55;
    }
    else
    {
      sub_13A38D0((__int64)&v60, (__int64)&v43);
      if ( v61 > 0x40 )
        sub_16A8F40(&v60);
      else
        v60 = ~v60 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v61);
      sub_16A7400(&v60);
      v32 = v61;
      v61 = 0;
      v63 = v32;
      v62 = v60;
      sub_13A3C50((__int64)v64, (__int64)&v62, (__int64)v59);
      sub_13A38D0((__int64)v65, (__int64)&v55);
      sub_13A3810((__int64)v66, (__int64)v65, (__int64)v64);
      sub_13A3610((__int64 *)&v55, v66);
      sub_135E100(v66);
      sub_135E100(v65);
      sub_135E100(v64);
      sub_135E100((__int64 *)&v62);
      sub_135E100((__int64 *)&v60);
      if ( !v37 )
      {
LABEL_22:
        sub_16A9F90(v66, &v47, &v41);
        sub_13A3610(v59, v66);
        sub_135E100(v66);
        if ( sub_13A39D0((__int64)v59, 0) )
        {
          sub_13A38D0((__int64)&v60, (__int64)&v45);
          if ( v61 > 0x40 )
            sub_16A8F40(&v60);
          else
            v60 = ~v60 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v61);
          sub_16A7400(&v60);
          v24 = v61;
          v61 = 0;
          v63 = v24;
          v62 = v60;
          sub_13A3A60((__int64)v64, (__int64)&v62, (__int64)v59);
          sub_13A38D0((__int64)v65, (__int64)&v57);
          sub_13A37A0((__int64)v66, (__int64)v65, (__int64)v64);
          sub_13A3610(&v57, v66);
          sub_135E100(v66);
          sub_135E100(v65);
          sub_135E100(v64);
          sub_135E100((__int64 *)&v62);
          sub_135E100((__int64 *)&v60);
          if ( !(_BYTE)v9 )
            goto LABEL_26;
          sub_13A38D0((__int64)&v60, (__int64)&v53);
          sub_16A7590(&v60, &v45);
          v35 = v61;
          v61 = 0;
          v63 = v35;
          v62 = v60;
          sub_13A3C50((__int64)v64, (__int64)&v62, (__int64)v59);
          v9 = (__int64 *)&v55;
          sub_13A38D0((__int64)v65, (__int64)&v55);
          sub_13A3810((__int64)v66, (__int64)v65, (__int64)v64);
        }
        else
        {
          sub_13A38D0((__int64)&v60, (__int64)&v45);
          if ( v61 > 0x40 )
            sub_16A8F40(&v60);
          else
            v60 = ~v60 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v61);
          sub_16A7400(&v60);
          v30 = v61;
          v61 = 0;
          v63 = v30;
          v62 = v60;
          sub_13A3C50((__int64)v64, (__int64)&v62, (__int64)v59);
          sub_13A38D0((__int64)v65, (__int64)&v55);
          sub_13A3810((__int64)v66, (__int64)v65, (__int64)v64);
          sub_13A3610((__int64 *)&v55, v66);
          sub_135E100(v66);
          sub_135E100(v65);
          sub_135E100(v64);
          sub_135E100((__int64 *)&v62);
          sub_135E100((__int64 *)&v60);
          if ( !(_BYTE)v9 )
            goto LABEL_26;
          sub_13A38D0((__int64)&v60, (__int64)&v53);
          sub_16A7590(&v60, &v45);
          v31 = v61;
          v61 = 0;
          v63 = v31;
          v62 = v60;
          sub_13A3A60((__int64)v64, (__int64)&v62, (__int64)v59);
          v9 = &v57;
          sub_13A38D0((__int64)v65, (__int64)&v57);
          sub_13A37A0((__int64)v66, (__int64)v65, (__int64)v64);
        }
        sub_13A3610(v9, v66);
        sub_135E100(v66);
        sub_135E100(v65);
        sub_135E100(v64);
        sub_135E100((__int64 *)&v62);
        sub_135E100((__int64 *)&v60);
LABEL_26:
        LOBYTE(v9) = (int)sub_16AEA10(&v57, &v55) > 0;
        sub_135E100(v59);
        sub_135E100(&v57);
        sub_135E100((__int64 *)&v55);
        sub_135E100((__int64 *)&v53);
        sub_135E100((__int64 *)&v51);
        goto LABEL_27;
      }
      sub_13A38D0((__int64)&v60, (__int64)&v51);
      sub_16A7590(&v60, &v43);
      v33 = v61;
      v61 = 0;
      v63 = v33;
      v62 = v60;
      sub_13A3A60((__int64)v64, (__int64)&v62, (__int64)v59);
      sub_13A38D0((__int64)v65, (__int64)&v57);
      sub_13A37A0((__int64)v66, (__int64)v65, (__int64)v64);
      v34 = &v57;
    }
    sub_13A3610(v34, v66);
    sub_135E100(v66);
    sub_135E100(v65);
    sub_135E100(v64);
    sub_135E100((__int64 *)&v62);
    sub_135E100((__int64 *)&v60);
    goto LABEL_22;
  }
LABEL_27:
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  sub_135E100(&v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  return (unsigned int)v9;
}
