// Function: sub_34428D0
// Address: 0x34428d0
//
__int64 __fastcall sub_34428D0(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4)
{
  __int64 v5; // r14
  unsigned int v6; // r15d
  __int64 v7; // r12
  __int64 v8; // r15
  unsigned int v9; // r13d
  bool v10; // dl
  _BYTE **v11; // r13
  bool v12; // dl
  unsigned int v13; // ebx
  int v14; // eax
  bool v15; // bl
  bool v16; // bl
  unsigned int v17; // r15d
  unsigned int v18; // r15d
  bool v19; // dl
  unsigned int v20; // r14d
  unsigned __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rdx
  unsigned __int8 *v24; // r8
  __int64 v25; // rax
  __int64 v26; // r9
  unsigned __int8 **v27; // rax
  __int64 v28; // r12
  __int64 v29; // rbx
  __int64 v30; // r15
  unsigned __int16 v31; // ax
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned int v34; // eax
  unsigned __int64 v35; // rdx
  unsigned __int8 *v36; // rax
  unsigned __int8 *v37; // rdx
  unsigned __int8 *v38; // r11
  __int64 v39; // rdx
  unsigned __int8 *v40; // r10
  unsigned __int8 **v41; // rdx
  __int64 v42; // rbx
  unsigned __int8 *v43; // rax
  __int64 v44; // r9
  unsigned __int8 *v45; // rdx
  unsigned __int8 *v46; // r13
  __int64 v47; // rdx
  unsigned __int8 *v48; // r12
  unsigned __int8 **v49; // rdx
  unsigned int v52; // eax
  unsigned int v54; // eax
  _QWORD *v55; // rdx
  __int64 v56; // [rsp-8h] [rbp-C8h]
  bool v57; // [rsp+8h] [rbp-B8h]
  int v58; // [rsp+8h] [rbp-B8h]
  __int64 v59; // [rsp+8h] [rbp-B8h]
  unsigned __int64 *v60; // [rsp+10h] [rbp-B0h]
  __int64 v61; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v62; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v63; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v64; // [rsp+18h] [rbp-A8h]
  __int64 v65; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v66; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v67; // [rsp+28h] [rbp-98h]
  _QWORD *v68; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v69; // [rsp+38h] [rbp-88h]
  _QWORD *v70; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v71; // [rsp+48h] [rbp-78h]
  unsigned __int64 v72; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v73; // [rsp+58h] [rbp-68h]
  unsigned __int64 v74; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v75; // [rsp+68h] [rbp-58h]
  __int64 v76; // [rsp+70h] [rbp-50h] BYREF
  char v77; // [rsp+78h] [rbp-48h]
  unsigned __int64 v78; // [rsp+80h] [rbp-40h] BYREF
  __int64 v79; // [rsp+88h] [rbp-38h]

  v5 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v6 = *(_DWORD *)(v5 + 32);
  v7 = v5 + 24;
  if ( v6 > 0x40 )
  {
    if ( v6 != (unsigned int)sub_C444A0(v5 + 24) )
      goto LABEL_3;
    return 0;
  }
  if ( !*(_QWORD *)(v5 + 24) )
    return 0;
LABEL_3:
  v8 = *(_QWORD *)(*(_QWORD *)a3 + 96LL);
  v9 = *(_DWORD *)(v8 + 32);
  v60 = (unsigned __int64 *)(v8 + 24);
  if ( v9 <= 0x40 )
    v10 = *(_QWORD *)(v8 + 24) == 0;
  else
    v10 = v9 == (unsigned int)sub_C444A0((__int64)v60);
  v11 = (_BYTE **)*a1;
  **(_BYTE **)*a1 &= v10;
  v12 = (int)sub_C49970(v5 + 24, v60) <= 0;
  *v11[1] |= v12;
  v13 = *(_DWORD *)(v5 + 32);
  if ( v13 <= 0x40 )
  {
    v15 = *(_QWORD *)(v5 + 24) == 1;
  }
  else
  {
    v57 = v12;
    v14 = sub_C444A0(v5 + 24);
    v12 = v57;
    v15 = v13 - 1 == v14;
  }
  v16 = v12 || v15;
  *v11[2] |= v16;
  *v11[3] &= v16;
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
  {
    v58 = *(_DWORD *)(v8 + 32);
    if ( v58 == (unsigned int)sub_C444A0((__int64)v60) )
      goto LABEL_10;
    goto LABEL_9;
  }
  if ( *(_QWORD *)(v8 + 24) )
LABEL_9:
    *v11[4] &= v16;
LABEL_10:
  v17 = *(_DWORD *)(v5 + 32);
  if ( v17 <= 0x40 )
  {
    _RDX = *(_QWORD *)(v5 + 24);
    v52 = 64;
    v67 = *(_DWORD *)(v5 + 32);
    __asm { tzcnt   rcx, rdx }
    v66 = _RDX;
    if ( _RDX )
      v52 = _RCX;
    if ( v17 <= v52 )
      v52 = v17;
    LODWORD(v59) = v52;
    goto LABEL_62;
  }
  v67 = *(_DWORD *)(v5 + 32);
  LODWORD(v59) = sub_C44590(v5 + 24);
  sub_C43780((__int64)&v66, (const void **)(v5 + 24));
  v17 = v67;
  if ( v67 <= 0x40 )
  {
LABEL_62:
    if ( (_DWORD)v59 == v17 )
      v66 = 0;
    else
      v66 >>= v59;
    goto LABEL_13;
  }
  sub_C482E0((__int64)&v66, v59);
LABEL_13:
  *v11[5] |= (_DWORD)v59 != 0;
  v18 = v67;
  if ( v67 <= 0x40 )
    v19 = v66 == 1;
  else
    v19 = v18 - 1 == (unsigned int)sub_C444A0((__int64)&v66);
  *v11[6] &= v19;
  v20 = *(_DWORD *)(v5 + 32);
  sub_C473B0((__int64)&v68, (__int64)&v66);
  v71 = 1;
  v70 = 0;
  v73 = 1;
  v72 = 0;
  LODWORD(v79) = v20;
  if ( v20 > 0x40 )
  {
    sub_C43690((__int64)&v78, -1, 1);
  }
  else
  {
    v21 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v20;
    if ( !v20 )
      v21 = 0;
    v78 = v21;
  }
  sub_C4BFE0((__int64)&v78, v7, &v70, &v72);
  if ( (unsigned int)v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  if ( (int)sub_C49970((__int64)v60, &v72) > 0 )
    sub_C46F20((__int64)&v70, 1u);
  if ( v16 )
  {
    if ( v69 <= 0x40 )
    {
      v68 = 0;
      v54 = v71;
      if ( v71 <= 0x40 )
        goto LABEL_70;
    }
    else
    {
      *v68 = 0;
      memset(v68 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v69 + 63) >> 6) - 8);
      v54 = v71;
      if ( v71 <= 0x40 )
      {
LABEL_70:
        v55 = (_QWORD *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v54);
        if ( !v54 )
          v55 = 0;
        v59 = 0xFFFFFFFFLL;
        v70 = v55;
        goto LABEL_26;
      }
    }
    *v70 = -1;
    memset(v70 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v71 + 63) >> 6) - 8);
    v59 = 0xFFFFFFFFLL;
    goto LABEL_26;
  }
  v59 = (unsigned int)v59;
LABEL_26:
  v22 = (__int64)v11[7];
  v24 = sub_34007B0(
          (__int64)v11[8],
          (__int64)&v68,
          (__int64)v11[9],
          *(_DWORD *)v11[10],
          *((_QWORD *)v11[10] + 1),
          0,
          a4,
          0);
  v25 = *(unsigned int *)(v22 + 8);
  v26 = v23;
  if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(v22 + 12) )
  {
    v63 = v24;
    v65 = v23;
    sub_C8D5F0(v22, (const void *)(v22 + 16), v25 + 1, 0x10u, (__int64)v24, v23);
    v25 = *(unsigned int *)(v22 + 8);
    v24 = v63;
    v26 = v65;
  }
  v27 = (unsigned __int8 **)(*(_QWORD *)v22 + 16 * v25);
  *v27 = v24;
  v27[1] = (unsigned __int8 *)v26;
  ++*(_DWORD *)(v22 + 8);
  v28 = (__int64)v11[12];
  v29 = (__int64)v11[11];
  v61 = (__int64)v11[8];
  v30 = (__int64)v11[9];
  v31 = *(_WORD *)v28;
  if ( *(_WORD *)v28 )
  {
    if ( v31 == 1 || (unsigned __int16)(v31 - 504) <= 7u )
      BUG();
    v33 = 16LL * (v31 - 1);
    v32 = *(_QWORD *)&byte_444C4A0[v33];
    LOBYTE(v33) = byte_444C4A0[v33 + 8];
  }
  else
  {
    v32 = sub_3007260(v28);
    v78 = v32;
    v79 = v33;
  }
  v76 = v32;
  v77 = v33;
  v34 = sub_CA1930(&v76);
  v75 = v34;
  if ( v34 > 0x40 )
  {
    sub_C43690((__int64)&v74, v59, 0);
  }
  else
  {
    v35 = 0;
    if ( v34 )
      v35 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v34) & v59;
    v74 = v35;
  }
  v36 = sub_34007B0(v61, (__int64)&v74, v30, *(_DWORD *)v28, *(_QWORD *)(v28 + 8), 0, a4, 0);
  v38 = v37;
  v39 = *(unsigned int *)(v29 + 8);
  v40 = v36;
  if ( v39 + 1 > (unsigned __int64)*(unsigned int *)(v29 + 12) )
  {
    v62 = v36;
    v64 = v38;
    sub_C8D5F0(v29, (const void *)(v29 + 16), v39 + 1, 0x10u, v39 + 1, v56);
    v39 = *(unsigned int *)(v29 + 8);
    v40 = v62;
    v38 = v64;
  }
  v41 = (unsigned __int8 **)(*(_QWORD *)v29 + 16 * v39);
  *v41 = v40;
  v41[1] = v38;
  ++*(_DWORD *)(v29 + 8);
  if ( v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  v42 = (__int64)v11[13];
  v43 = sub_34007B0(
          (__int64)v11[8],
          (__int64)&v70,
          (__int64)v11[9],
          *(_DWORD *)v11[10],
          *((_QWORD *)v11[10] + 1),
          0,
          a4,
          0);
  v46 = v45;
  v47 = *(unsigned int *)(v42 + 8);
  v48 = v43;
  if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(v42 + 12) )
  {
    sub_C8D5F0(v42, (const void *)(v42 + 16), v47 + 1, 0x10u, v47 + 1, v44);
    v47 = *(unsigned int *)(v42 + 8);
  }
  v49 = (unsigned __int8 **)(*(_QWORD *)v42 + 16 * v47);
  *v49 = v48;
  v49[1] = v46;
  ++*(_DWORD *)(v42 + 8);
  if ( v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  if ( v71 > 0x40 && v70 )
    j_j___libc_free_0_0((unsigned __int64)v70);
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0((unsigned __int64)v68);
  if ( v67 > 0x40 )
  {
    if ( v66 )
      j_j___libc_free_0_0(v66);
  }
  return 1;
}
