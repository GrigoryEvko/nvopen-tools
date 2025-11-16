// Function: sub_2473860
// Address: 0x2473860
//
void __fastcall sub_2473860(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v7; // r15
  int v8; // ebx
  __int64 v9; // r15
  int v10; // eax
  unsigned int v11; // ebx
  int v12; // r13d
  unsigned int v13; // ecx
  unsigned int v14; // edx
  unsigned int v16; // eax
  unsigned int v18; // esi
  __int64 v19; // rdi
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rdx
  unsigned __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // r9
  __int64 v29; // rdx
  unsigned __int64 v30; // r8
  __int64 *v31; // rdi
  __int64 v32; // rsi
  unsigned __int8 *v33; // r12
  unsigned __int8 *v34; // rbx
  __int64 (__fastcall *v35)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v36; // r15
  __int64 v37; // rax
  unsigned __int64 v38; // r12
  char *v39; // rbx
  __int64 v40; // rdx
  unsigned int v41; // esi
  unsigned int v42; // eax
  unsigned int v43; // edx
  unsigned int v45; // r12d
  unsigned int v47; // esi
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rax
  unsigned int v50; // eax
  unsigned int v53; // [rsp+18h] [rbp-138h]
  __int64 v54; // [rsp+18h] [rbp-138h]
  __int64 v55; // [rsp+18h] [rbp-138h]
  unsigned __int64 v57; // [rsp+30h] [rbp-120h] BYREF
  unsigned int v58; // [rsp+38h] [rbp-118h]
  char v59; // [rsp+50h] [rbp-100h]
  char v60; // [rsp+51h] [rbp-FFh]
  unsigned __int64 v61; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v62; // [rsp+68h] [rbp-E8h]
  __int16 v63; // [rsp+80h] [rbp-D0h]
  unsigned __int64 v64; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v65; // [rsp+98h] [rbp-B8h]
  _BYTE v66[40]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v67; // [rsp+C8h] [rbp-88h]
  __int64 v68; // [rsp+D0h] [rbp-80h]
  __int64 v69; // [rsp+E0h] [rbp-70h]
  __int64 v70; // [rsp+E8h] [rbp-68h]

  v4 = a2;
  v5 = a1;
  v7 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
  {
    if ( *(_BYTE *)a3 != 17 )
    {
      v33 = (unsigned __int8 *)sub_AD64C0(v7, 1, 0);
      sub_23D0AB0((__int64)&v64, a2, 0, 0, 0);
      goto LABEL_34;
    }
    v42 = *(_DWORD *)(a3 + 32);
    LODWORD(v65) = v42;
    if ( v42 > 0x40 )
    {
      sub_C43690((__int64)&v64, 1, 0);
      v42 = *(_DWORD *)(a3 + 32);
      if ( v42 > 0x40 )
      {
        v50 = sub_C44590(a3 + 24);
        v43 = v65;
        v45 = v50;
        goto LABEL_55;
      }
      v43 = v65;
    }
    else
    {
      v43 = v42;
      v64 = 1;
    }
    _RCX = *(_QWORD *)(a3 + 24);
    v45 = 64;
    __asm { tzcnt   rsi, rcx }
    if ( _RCX )
      v45 = _RSI;
    if ( v42 <= v45 )
      v45 = v42;
LABEL_55:
    v62 = v43;
    if ( v43 > 0x40 )
    {
      sub_C43780((__int64)&v61, (const void **)&v64);
      v43 = v62;
      if ( v62 > 0x40 )
      {
        sub_C47690((__int64 *)&v61, v45);
        v47 = v65;
LABEL_62:
        if ( v47 > 0x40 && v64 )
          j_j___libc_free_0_0(v64);
        v33 = (unsigned __int8 *)sub_AD8D80(v7, (__int64)&v61);
        if ( v62 > 0x40 && v61 )
          j_j___libc_free_0_0(v61);
        goto LABEL_33;
      }
      v47 = v65;
    }
    else
    {
      v47 = v43;
      v61 = v64;
    }
    v48 = 0;
    if ( v43 != v45 )
      v48 = v61 << v45;
    v49 = v48 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v43);
    if ( !v43 )
      v49 = 0;
    v61 = v49;
    goto LABEL_62;
  }
  v8 = *(_DWORD *)(v7 + 32);
  v9 = *(_QWORD *)(v7 + 24);
  v64 = (unsigned __int64)v66;
  v65 = 0x1000000000LL;
  v10 = v8;
  if ( !v8 )
  {
    v31 = (__int64 *)v66;
    v32 = 0;
    goto LABEL_31;
  }
  v11 = 0;
  v12 = v10;
  do
  {
    while ( 1 )
    {
      v25 = sub_AD69F0((unsigned __int8 *)a3, v11);
      v26 = v25;
      if ( *(_BYTE *)v25 == 17 )
        break;
      v27 = sub_AD64C0(v9, 1, 0);
      v29 = (unsigned int)v65;
      v30 = (unsigned int)v65 + 1LL;
      if ( v30 > HIDWORD(v65) )
      {
        v55 = v27;
        sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 8u, v30, v28);
        v29 = (unsigned int)v65;
        v27 = v55;
      }
      ++v11;
      *(_QWORD *)(v64 + 8 * v29) = v27;
      LODWORD(v65) = v65 + 1;
      if ( v12 == v11 )
        goto LABEL_30;
    }
    v13 = *(_DWORD *)(v25 + 32);
    v62 = v13;
    if ( v13 > 0x40 )
    {
      sub_C43690((__int64)&v61, 1, 0);
      v13 = *(_DWORD *)(v26 + 32);
      if ( v13 > 0x40 )
      {
        v16 = sub_C44590(v26 + 24);
        v14 = v62;
        v58 = v62;
        if ( v62 <= 0x40 )
          goto LABEL_11;
        goto LABEL_43;
      }
      v14 = v62;
    }
    else
    {
      v61 = 1;
      v14 = v13;
    }
    _RSI = *(_QWORD *)(v26 + 24);
    v16 = 64;
    v58 = v14;
    __asm { tzcnt   rdi, rsi }
    if ( _RSI )
      v16 = _RDI;
    if ( v16 > v13 )
      v16 = v13;
    if ( v14 <= 0x40 )
    {
LABEL_11:
      v18 = v14;
      v57 = v61;
      goto LABEL_12;
    }
LABEL_43:
    v53 = v16;
    sub_C43780((__int64)&v57, (const void **)&v61);
    v14 = v58;
    v16 = v53;
    if ( v58 > 0x40 )
    {
      sub_C47690((__int64 *)&v57, v53);
      v18 = v62;
      goto LABEL_17;
    }
    v18 = v62;
LABEL_12:
    v19 = 0;
    if ( v16 != v14 )
      v19 = v57 << v16;
    v20 = v19 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
    if ( !v14 )
      v20 = 0;
    v57 = v20;
LABEL_17:
    if ( v18 > 0x40 && v61 )
      j_j___libc_free_0_0(v61);
    v21 = sub_AD8D80(v9, (__int64)&v57);
    v23 = (unsigned int)v65;
    v24 = (unsigned int)v65 + 1LL;
    if ( v24 > HIDWORD(v65) )
    {
      v54 = v21;
      sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 8u, v24, v22);
      v23 = (unsigned int)v65;
      v21 = v54;
    }
    *(_QWORD *)(v64 + 8 * v23) = v21;
    LODWORD(v65) = v65 + 1;
    if ( v58 > 0x40 && v57 )
      j_j___libc_free_0_0(v57);
    ++v11;
  }
  while ( v12 != v11 );
LABEL_30:
  v5 = a1;
  v4 = a2;
  v31 = (__int64 *)v64;
  v32 = (unsigned int)v65;
LABEL_31:
  v33 = (unsigned __int8 *)sub_AD3730(v31, v32);
  if ( (_BYTE *)v64 != v66 )
    _libc_free(v64);
LABEL_33:
  sub_23D0AB0((__int64)&v64, v4, 0, 0, 0);
LABEL_34:
  v60 = 1;
  v57 = (unsigned __int64)"msprop_mul_cst";
  v59 = 3;
  v34 = (unsigned __int8 *)sub_246F3F0(v5, a4);
  v35 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v69 + 32LL);
  if ( v35 != sub_9201A0 )
  {
    v36 = v35(v69, 17u, v34, v33, 0, 0);
    goto LABEL_39;
  }
  if ( *v34 > 0x15u || *v33 > 0x15u )
    goto LABEL_45;
  v36 = (unsigned __int8)sub_AC47B0(17) ? sub_AD5570(17, (__int64)v34, v33, 0, 0) : sub_AABE40(0x11u, v34, v33);
LABEL_39:
  if ( !v36 )
  {
LABEL_45:
    v63 = 257;
    v36 = sub_B504D0(17, (__int64)v34, (__int64)v33, (__int64)&v61, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, __int64))(*(_QWORD *)v70 + 16LL))(
      v70,
      v36,
      &v57,
      v67,
      v68);
    v38 = v64;
    v39 = (char *)(v64 + 16LL * (unsigned int)v65);
    if ( (char *)v64 != v39 )
    {
      do
      {
        v40 = *(_QWORD *)(v38 + 8);
        v41 = *(_DWORD *)v38;
        v38 += 16LL;
        sub_B99FD0(v36, v41, v40);
      }
      while ( v39 != (char *)v38 );
    }
  }
  sub_246EF60(v5, v4, v36);
  v37 = sub_246EE10(v5, a4);
  sub_246F1C0(v5, v4, v37);
  sub_F94A20(&v64, v4);
}
