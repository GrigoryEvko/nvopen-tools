// Function: sub_17D7E80
// Address: 0x17d7e80
//
unsigned __int64 __fastcall sub_17D7E80(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        const char *a4,
        double a5,
        double a6,
        double a7)
{
  __int64 v8; // r13
  __int64 v10; // rbx
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // ecx
  unsigned int v14; // r12d
  unsigned int v15; // edx
  unsigned __int64 v16; // rax
  __int64 v17; // r12
  __int128 v18; // rdi
  __int64 *v19; // rdi
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v25; // rax
  __int64 *v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // ebx
  int v32; // r13d
  __int64 v33; // r14
  unsigned int v34; // edx
  unsigned int v35; // ecx
  unsigned int v37; // r8d
  unsigned int v39; // edx
  const char *v40; // rax
  __int64 v41; // r8
  int v42; // r9d
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r15
  __int64 v46; // r8
  int v47; // r9d
  __int64 v48; // rax
  __int64 *v49; // rdi
  __int64 v50; // rsi
  unsigned int v51; // eax
  __int64 *v54; // [rsp+0h] [rbp-130h]
  unsigned int v55; // [rsp+20h] [rbp-110h]
  __int64 v56; // [rsp+20h] [rbp-110h]
  __int64 v57; // [rsp+20h] [rbp-110h]
  unsigned __int64 v59; // [rsp+30h] [rbp-100h] BYREF
  unsigned int v60; // [rsp+38h] [rbp-F8h]
  char v61; // [rsp+40h] [rbp-F0h]
  char v62; // [rsp+41h] [rbp-EFh]
  unsigned __int64 v63; // [rsp+50h] [rbp-E0h] BYREF
  unsigned int v64; // [rsp+58h] [rbp-D8h]
  __int16 v65; // [rsp+60h] [rbp-D0h]
  __int64 *v66; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+78h] [rbp-B8h]
  _QWORD v68[22]; // [rsp+80h] [rbp-B0h] BYREF

  v8 = a1;
  v10 = *(_QWORD *)a3;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) != 16 )
  {
    if ( *(_BYTE *)(a3 + 16) != 13 )
    {
      v17 = sub_15A0680(v10, 1, 0);
      goto LABEL_17;
    }
    v11 = *(_DWORD *)(a3 + 32);
    LODWORD(v67) = v11;
    if ( v11 <= 0x40 )
    {
      v13 = v11;
      v66 = (__int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & 1);
    }
    else
    {
      sub_16A4EF0((__int64)&v66, 1, 0);
      v11 = *(_DWORD *)(a3 + 32);
      if ( v11 > 0x40 )
      {
        v12 = sub_16A58A0(a3 + 24);
        v13 = v67;
        v14 = v12;
        v64 = v67;
        if ( (unsigned int)v67 <= 0x40 )
        {
LABEL_6:
          v15 = v13;
          v63 = (unsigned __int64)v66;
          goto LABEL_7;
        }
LABEL_63:
        sub_16A4FD0((__int64)&v63, (const void **)&v66);
        v13 = v64;
        if ( v64 > 0x40 )
        {
          sub_16A7DC0((__int64 *)&v63, v14);
          v15 = v67;
LABEL_10:
          if ( v15 > 0x40 && v66 )
            j_j___libc_free_0_0(v66);
          v17 = sub_15A1070(v10, (__int64)&v63);
          if ( v64 > 0x40 && v63 )
            j_j___libc_free_0_0(v63);
          goto LABEL_17;
        }
        v15 = v67;
LABEL_7:
        v16 = 0;
        if ( v13 != v14 )
          v16 = (v63 << v14) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13);
        v63 = v16;
        goto LABEL_10;
      }
      v13 = v67;
    }
    _RDX = *(_QWORD *)(a3 + 24);
    v14 = 64;
    v64 = v13;
    __asm { tzcnt   rsi, rdx }
    if ( _RDX )
      v14 = _RSI;
    if ( v11 <= v14 )
      v14 = v11;
    if ( v13 <= 0x40 )
      goto LABEL_6;
    goto LABEL_63;
  }
  v29 = *(_QWORD *)(v10 + 32);
  v30 = **(_QWORD **)(v10 + 16);
  v66 = v68;
  v67 = 0x1000000000LL;
  if ( !(_DWORD)v29 )
  {
    v49 = v68;
    v50 = 0;
    goto LABEL_51;
  }
  v54 = a2;
  v31 = 0;
  v32 = v29;
  v33 = v30;
  do
  {
    while ( 1 )
    {
      v44 = sub_15A0A60(a3, v31);
      v45 = v44;
      if ( *(_BYTE *)(v44 + 16) == 13 )
        break;
      v46 = sub_15A0680(v33, 1, 0);
      v48 = (unsigned int)v67;
      if ( (unsigned int)v67 >= HIDWORD(v67) )
      {
        v57 = v46;
        sub_16CD150((__int64)&v66, v68, 0, 8, v46, v47);
        v48 = (unsigned int)v67;
        v46 = v57;
      }
      ++v31;
      v66[v48] = v46;
      LODWORD(v67) = v67 + 1;
      if ( v32 == v31 )
        goto LABEL_50;
    }
    v34 = *(_DWORD *)(v44 + 32);
    v64 = v34;
    if ( v34 > 0x40 )
    {
      sub_16A4EF0((__int64)&v63, 1, 0);
      v34 = *(_DWORD *)(v45 + 32);
      if ( v34 > 0x40 )
      {
        v51 = sub_16A58A0(v45 + 24);
        v35 = v64;
        v37 = v51;
        v60 = v64;
        if ( v64 <= 0x40 )
          goto LABEL_33;
        goto LABEL_55;
      }
      v35 = v64;
    }
    else
    {
      v63 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v34) & 1;
      v35 = v34;
    }
    _RAX = *(_QWORD *)(v45 + 24);
    v37 = 64;
    v60 = v35;
    __asm { tzcnt   rsi, rax }
    if ( _RAX )
      v37 = _RSI;
    if ( v34 <= v37 )
      v37 = v34;
    if ( v35 <= 0x40 )
    {
LABEL_33:
      v39 = v35;
      v59 = v63;
      goto LABEL_34;
    }
LABEL_55:
    v55 = v37;
    sub_16A4FD0((__int64)&v59, (const void **)&v63);
    v35 = v60;
    v37 = v55;
    if ( v60 > 0x40 )
    {
      sub_16A7DC0((__int64 *)&v59, v55);
      v39 = v64;
      goto LABEL_37;
    }
    v39 = v64;
LABEL_34:
    v40 = 0;
    if ( v35 != v37 )
      v40 = (const char *)((v59 << v37) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v35));
    v59 = (unsigned __int64)v40;
LABEL_37:
    if ( v39 > 0x40 && v63 )
      j_j___libc_free_0_0(v63);
    v41 = sub_15A1070(v33, (__int64)&v59);
    v43 = (unsigned int)v67;
    if ( (unsigned int)v67 >= HIDWORD(v67) )
    {
      v56 = v41;
      sub_16CD150((__int64)&v66, v68, 0, 8, v41, v42);
      v43 = (unsigned int)v67;
      v41 = v56;
    }
    v66[v43] = v41;
    LODWORD(v67) = v67 + 1;
    if ( v60 > 0x40 && v59 )
      j_j___libc_free_0_0(v59);
    ++v31;
  }
  while ( v32 != v31 );
LABEL_50:
  v8 = a1;
  a2 = v54;
  v49 = v66;
  v50 = (unsigned int)v67;
LABEL_51:
  v17 = sub_15A01B0(v49, v50);
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
LABEL_17:
  sub_17CE510((__int64)&v66, (__int64)a2, 0, 0, 0);
  *((_QWORD *)&v18 + 1) = a4;
  *(_QWORD *)&v18 = v8;
  v62 = 1;
  v59 = (unsigned __int64)"msprop_mul_cst";
  v61 = 3;
  v19 = sub_17D4DA0(v18);
  if ( *((_BYTE *)v19 + 16) > 0x10u || *(_BYTE *)(v17 + 16) > 0x10u )
  {
    v65 = 257;
    v25 = sub_15FB440(15, v19, v17, (__int64)&v63, 0);
    v20 = v25;
    if ( v67 )
    {
      v26 = (__int64 *)v68[0];
      sub_157E9D0(v67 + 40, v25);
      v27 = *(_QWORD *)(v20 + 24);
      v28 = *v26;
      *(_QWORD *)(v20 + 32) = v26;
      v28 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v20 + 24) = v28 | v27 & 7;
      *(_QWORD *)(v28 + 8) = v20 + 24;
      *v26 = *v26 & 7 | (v20 + 24);
    }
    sub_164B780(v20, (__int64 *)&v59);
    sub_12A86E0((__int64 *)&v66, v20);
  }
  else
  {
    v20 = sub_15A2C20(v19, v17, 0, 0, a5, a6, a7);
  }
  sub_17D4920(v8, a2, v20);
  v23 = sub_17D4880(v8, a4, v21, v22);
  sub_17D4B80(v8, (__int64)a2, v23);
  return sub_17CD270((__int64 *)&v66);
}
