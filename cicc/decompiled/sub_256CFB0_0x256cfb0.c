// Function: sub_256CFB0
// Address: 0x256cfb0
//
__int64 __fastcall sub_256CFB0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7,
        int *a8,
        __int64 a9)
{
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // rdx
  unsigned __int64 v15; // r8
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 **v18; // r10
  __int64 v19; // r8
  _QWORD *v20; // r9
  int v21; // eax
  unsigned __int64 v22; // rsi
  unsigned int v23; // ecx
  int v24; // r13d
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 *v27; // rdx
  unsigned int v28; // eax
  int v29; // eax
  __int64 *v30; // rdi
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rdx
  int v42; // eax
  __int64 i; // r14
  unsigned __int8 *v44; // rax
  __int64 v45; // r8
  __int64 v46; // r9
  unsigned int v47; // eax
  __int64 *v48; // rax
  __int64 *v49; // rdx
  int v50; // [rsp+18h] [rbp-148h]
  __int64 v51; // [rsp+18h] [rbp-148h]
  int v52; // [rsp+18h] [rbp-148h]
  unsigned __int64 v53; // [rsp+20h] [rbp-140h]
  _QWORD *v54; // [rsp+20h] [rbp-140h]
  __int64 v55; // [rsp+20h] [rbp-140h]
  _QWORD *v56; // [rsp+20h] [rbp-140h]
  __int64 v57; // [rsp+28h] [rbp-138h]
  __int64 v58; // [rsp+28h] [rbp-138h]
  _QWORD **v59; // [rsp+28h] [rbp-138h]
  __int64 v60; // [rsp+28h] [rbp-138h]
  __int64 v61; // [rsp+38h] [rbp-128h]
  __int64 **v62; // [rsp+38h] [rbp-128h]
  __int64 v67; // [rsp+60h] [rbp-100h] BYREF
  __int64 v68; // [rsp+68h] [rbp-F8h]
  void *base; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v70; // [rsp+78h] [rbp-E8h]
  _BYTE v71[48]; // [rsp+80h] [rbp-E0h] BYREF
  __int64 *v72; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v73; // [rsp+B8h] [rbp-A8h]
  _BYTE v74[48]; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 *v75; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v76; // [rsp+F8h] [rbp-68h]
  _BYTE v77[96]; // [rsp+100h] [rbp-60h] BYREF

  v11 = *(_QWORD *)(*(_QWORD *)(a2 + 208) + 104LL);
  v75 = (__int64 *)sub_9208B0(v11, a9);
  v76 = v14;
  v15 = ((unsigned __int64)v75 + 7) >> 3;
  if ( (_BYTE)v14 )
    v15 = 0x7FFFFFFF;
  if ( *(_QWORD *)(a7 + 88) )
  {
    v16 = *(__int64 **)(a7 + 72);
    LOBYTE(v76) = 0;
    base = v71;
    v70 = 0x600000000LL;
    v75 = (__int64 *)(a7 + 56);
    LOBYTE(v73) = 0;
  }
  else
  {
    v16 = *(__int64 **)a7;
    v17 = *(unsigned int *)(a7 + 8);
    LOBYTE(v76) = 1;
    base = v71;
    v70 = 0x600000000LL;
    v75 = &v16[v17];
    LOBYTE(v73) = 1;
  }
  v72 = v16;
  v53 = v15;
  sub_255DF80((__int64)&base, (__int64)&v72, (__int64)&v75, v12, v15, v13);
  v18 = &v75;
  v19 = v53;
  if ( (unsigned int)v70 > 1uLL )
  {
    qsort(base, (8LL * (unsigned int)v70) >> 3, 8u, (__compar_fn_t)sub_F8E3D0);
    v18 = &v75;
    v19 = v53;
  }
  if ( *(_BYTE *)(a9 + 8) != 17
    || (_BYTE)a5 != 1
    || !a4
    || *(_BYTE *)a4 > 0x15u
    || a9 != *(_QWORD *)(a4 + 8)
    || (v58 = v19,
        v33 = sub_9C6480(v11, *(_QWORD *)(a9 + 24)),
        v19 = v58,
        v18 = &v75,
        v76 = v34,
        v75 = (__int64 *)v33,
        (_BYTE)v34) )
  {
    v75 = (__int64 *)v77;
    v20 = base;
    v76 = 0x300000000LL;
    v21 = v70;
    if ( (unsigned int)v70 > 3 )
    {
      v54 = base;
      v57 = v19;
      v50 = v70;
      sub_C8D5F0((__int64)&v75, v77, (unsigned int)v70, 0x10u, v19, (__int64)base);
      v23 = v76;
      v22 = HIDWORD(v76);
      v18 = &v75;
      v19 = v57;
      v20 = v54;
      v21 = v50;
    }
    else
    {
      if ( !(_DWORD)v70 )
        goto LABEL_18;
      v22 = 3;
      v23 = 0;
    }
    v24 = 0;
    while ( 1 )
    {
      v25 = v23;
      v26 = v20[v24];
      if ( v23 >= v22 )
      {
        if ( v22 < (unsigned __int64)v23 + 1 )
        {
          v52 = v21;
          v56 = v20;
          v60 = v19;
          v62 = v18;
          sub_C8D5F0((__int64)v18, v77, v23 + 1LL, 0x10u, v19, (__int64)v20);
          v25 = (unsigned int)v76;
          v21 = v52;
          v20 = v56;
          v19 = v60;
          v18 = v62;
        }
        v32 = &v75[2 * v25];
        *v32 = v26;
        v32[1] = v19;
        LODWORD(v76) = v76 + 1;
      }
      else
      {
        v27 = &v75[2 * v23];
        if ( v27 )
        {
          *v27 = v26;
          v27[1] = v19;
          v23 = v76;
        }
        LODWORD(v76) = v23 + 1;
      }
      if ( v21 == ++v24 )
        break;
      v23 = v76;
      v22 = HIDWORD(v76);
    }
LABEL_18:
    v28 = sub_256C1D0(a1 + 88, a2, (__int64)v18, a3, a4, a5, a6, a9, 0);
    v29 = sub_250C0B0(*a8, v28);
    v30 = v75;
    *a8 = v29;
    if ( v30 == (__int64 *)v77 )
      goto LABEL_20;
    goto LABEL_19;
  }
  v59 = *(_QWORD ***)(a9 + 24);
  v35 = sub_9C6480(v11, (__int64)v59);
  v76 = v36;
  v75 = (__int64 *)v35;
  v51 = v35;
  v55 = sub_BCB2D0(*v59);
  if ( *(_QWORD *)(a7 + 88) )
  {
    v40 = *(_QWORD *)(a7 + 72);
    LOBYTE(v76) = 0;
    v72 = (__int64 *)v74;
    v73 = 0x600000000LL;
    v75 = (__int64 *)(a7 + 56);
    LOBYTE(v68) = 0;
  }
  else
  {
    v40 = *(_QWORD *)a7;
    v41 = *(unsigned int *)(a7 + 8);
    LOBYTE(v76) = 1;
    v72 = (__int64 *)v74;
    v73 = 0x600000000LL;
    v75 = (__int64 *)(v40 + 8 * v41);
    LOBYTE(v68) = 1;
  }
  v67 = v40;
  sub_255DF80((__int64)&v72, (__int64)&v67, (__int64)&v75, v37, v38, v39);
  v42 = *(_DWORD *)(a9 + 32);
  if ( v42 )
  {
    v61 = (unsigned int)(v42 - 1);
    for ( i = 0; ; ++i )
    {
      v44 = (unsigned __int8 *)sub_AD64C0(v55, i, 0);
      v67 = sub_AD5840(a4, v44, 0);
      LOBYTE(v68) = 1;
      sub_25535D0((__int64)&v75, v72, (unsigned int)v73, v51, v45, v46);
      v47 = sub_256C1D0(a1 + 88, a2, (__int64)&v75, a3, v67, v68, a6, (__int64)v59, 0);
      *a8 = sub_250C0B0(*a8, v47);
      if ( v75 != (__int64 *)v77 )
        _libc_free((unsigned __int64)v75);
      v48 = v72;
      v49 = &v72[(unsigned int)v73];
      if ( v49 != v72 )
      {
        do
          *v48++ += v51;
        while ( v48 != v49 );
      }
      if ( v61 == i )
        break;
    }
  }
  v30 = v72;
  if ( v72 != (__int64 *)v74 )
LABEL_19:
    _libc_free((unsigned __int64)v30);
LABEL_20:
  if ( base != v71 )
    _libc_free((unsigned __int64)base);
  return 1;
}
