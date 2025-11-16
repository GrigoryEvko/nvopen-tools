// Function: sub_B03CA0
// Address: 0xb03ca0
//
__int64 __fastcall sub_B03CA0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8,
        __int64 a9,
        __int64 a10,
        unsigned __int64 a11,
        __int64 a12,
        __int64 a13,
        unsigned int a14,
        char a15)
{
  int v15; // r13d
  _QWORD *v16; // r12
  __int64 v17; // rbx
  unsigned int v18; // r14d
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rdx
  unsigned int v22; // eax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 *v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 *v29; // rdx
  unsigned int v30; // eax
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 *v33; // rdx
  unsigned int v34; // eax
  __int64 v35; // rcx
  int v36; // eax
  int v37; // r8d
  __int64 *v38; // r10
  int v39; // r12d
  int v40; // r13d
  unsigned int i; // ebx
  __int64 *v42; // r15
  __int64 v43; // rsi
  char v44; // al
  unsigned int v45; // ebx
  __int64 result; // rax
  __int64 v47; // r15
  __int64 v48; // rdi
  __int64 *v49; // [rsp+8h] [rbp-F8h]
  int v50; // [rsp+10h] [rbp-F0h]
  __int64 v51; // [rsp+10h] [rbp-F0h]
  __int64 v52; // [rsp+18h] [rbp-E8h]
  __int64 v53; // [rsp+20h] [rbp-E0h]
  int v54; // [rsp+28h] [rbp-D8h]
  int v58; // [rsp+5Ch] [rbp-A4h] BYREF
  __int64 v59; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v60; // [rsp+68h] [rbp-98h] BYREF
  __int64 v61; // [rsp+70h] [rbp-90h] BYREF
  __int64 v62; // [rsp+78h] [rbp-88h] BYREF
  __int64 v63; // [rsp+80h] [rbp-80h] BYREF
  __int64 v64; // [rsp+88h] [rbp-78h] BYREF
  __int64 v65; // [rsp+90h] [rbp-70h]
  unsigned __int64 v66; // [rsp+98h] [rbp-68h] BYREF
  __int64 v67; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v68; // [rsp+A8h] [rbp-58h]
  __int64 v69; // [rsp+B0h] [rbp-50h]
  __int64 v70; // [rsp+B8h] [rbp-48h]
  __int64 v71; // [rsp+C0h] [rbp-40h]

  v15 = a4;
  v16 = a1;
  v17 = a6;
  v18 = a14;
  if ( a14 )
    goto LABEL_37;
  LODWORD(v63) = a4;
  v19 = *a1;
  v65 = a6;
  v61 = a2;
  v62 = a3;
  v64 = a5;
  v66 = __PAIR64__(a8, a7);
  v67 = a9;
  v68 = a10;
  v69 = a11;
  v70 = a12;
  v71 = a13;
  v53 = v19;
  v52 = *(_QWORD *)(v19 + 1600);
  v50 = *(_DWORD *)(v19 + 1616);
  if ( !v50 )
    goto LABEL_36;
  v58 = 0;
  v59 = a10;
  if ( a10 && *(_BYTE *)a10 == 1 )
  {
    v20 = *(_QWORD *)(a10 + 136);
    v21 = *(__int64 **)(v20 + 24);
    v22 = *(_DWORD *)(v20 + 32);
    if ( v22 > 0x40 )
    {
      v23 = *v21;
    }
    else
    {
      v23 = 0;
      if ( v22 )
        v23 = (__int64)((_QWORD)v21 << (64 - (unsigned __int8)v22)) >> (64 - (unsigned __int8)v22);
    }
    v60 = v23;
    v58 = sub_AF6E60(&v58, &v60);
  }
  else
  {
    v58 = sub_AF7970(&v58, &v59);
  }
  v59 = v69;
  if ( v69 && *(_BYTE *)v69 == 1 )
  {
    v24 = *(_QWORD *)(v69 + 136);
    v25 = *(__int64 **)(v24 + 24);
    v26 = *(_DWORD *)(v24 + 32);
    if ( v26 > 0x40 )
    {
      v27 = *v25;
    }
    else
    {
      v27 = 0;
      if ( v26 )
        v27 = (__int64)((_QWORD)v25 << (64 - (unsigned __int8)v26)) >> (64 - (unsigned __int8)v26);
    }
    v60 = v27;
    v58 = sub_AF6E60(&v58, &v60);
  }
  else
  {
    v58 = sub_AF7970(&v58, &v59);
  }
  v59 = v70;
  if ( v70 && *(_BYTE *)v70 == 1 )
  {
    v28 = *(_QWORD *)(v70 + 136);
    v29 = *(__int64 **)(v28 + 24);
    v30 = *(_DWORD *)(v28 + 32);
    if ( v30 > 0x40 )
    {
      v31 = *v29;
    }
    else
    {
      v31 = 0;
      if ( v30 )
        v31 = (__int64)((_QWORD)v29 << (64 - (unsigned __int8)v30)) >> (64 - (unsigned __int8)v30);
    }
    v60 = v31;
    v58 = sub_AF6E60(&v58, &v60);
  }
  else
  {
    v58 = sub_AF7970(&v58, &v59);
  }
  v59 = v71;
  if ( v71 && *(_BYTE *)v71 == 1 )
  {
    v32 = *(_QWORD *)(v71 + 136);
    v33 = *(__int64 **)(v32 + 24);
    v34 = *(_DWORD *)(v32 + 32);
    if ( v34 > 0x40 )
    {
      v35 = *v33;
    }
    else
    {
      v35 = 0;
      if ( v34 )
        v35 = (__int64)((_QWORD)v33 << (64 - (unsigned __int8)v34)) >> (64 - (unsigned __int8)v34);
    }
    v60 = v35;
    v58 = sub_AF6E60(&v58, &v60);
  }
  else
  {
    v58 = sub_AF7970(&v58, &v59);
  }
  v36 = sub_AF95C0(&v58, &v61, &v62, (int *)&v63, &v64, &v67, (int *)&v66 + 1);
  v37 = v50;
  v38 = &v61;
  v51 = v17;
  v54 = v37 - 1;
  v39 = v15;
  v40 = 1;
  for ( i = (v37 - 1) & v36; ; i = v54 & v45 )
  {
    v42 = (__int64 *)(v52 + 8LL * i);
    v43 = *v42;
    if ( *v42 == -8192 )
      goto LABEL_32;
    if ( v43 == -4096 )
      goto LABEL_49;
    v49 = v38;
    v44 = sub_AF1900(v38, v43);
    v38 = v49;
    if ( v44 )
      break;
    v43 = *v42;
LABEL_32:
    if ( v43 == -4096 )
    {
LABEL_49:
      v15 = v39;
      v17 = v51;
      v16 = a1;
      v18 = 0;
      goto LABEL_36;
    }
    v45 = v40 + i;
    ++v40;
  }
  v15 = v39;
  v17 = v51;
  v16 = a1;
  v18 = 0;
  if ( v42 == (__int64 *)(*(_QWORD *)(v53 + 1600) + 8LL * *(unsigned int *)(v53 + 1616)) || (result = *v42) == 0 )
  {
LABEL_36:
    result = 0;
    if ( a15 )
    {
LABEL_37:
      v61 = a3;
      v62 = a5;
      v63 = a2;
      v64 = a9;
      v65 = a10;
      v66 = a11;
      v67 = a12;
      v68 = a13;
      v47 = *v16 + 1592LL;
      v48 = sub_B97910(48, 8, v18);
      if ( v48 )
        sub_AF2B40(v48, (int)v16, v18, v15, v17, a7, a8, (__int64)&v61, 8);
      return sub_B03BC0(v48, v18, v47);
    }
  }
  return result;
}
