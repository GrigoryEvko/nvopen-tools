// Function: sub_175FF30
// Address: 0x175ff30
//
__int64 __fastcall sub_175FF30(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        _QWORD ***a4,
        int a5,
        int a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v14; // r14
  unsigned int v15; // edx
  unsigned int v16; // r12d
  __int64 v17; // r15
  _BYTE *v18; // r14
  __int64 v19; // rax
  _QWORD *v20; // r13
  _QWORD *v21; // rax
  int v22; // r8d
  int v23; // r9d
  int v24; // edx
  __int64 v25; // rdi
  __int64 v26; // r13
  __int64 v27; // rax
  unsigned int v28; // ecx
  _BYTE *v29; // r13
  __int64 v30; // r14
  _QWORD *v32; // rdi
  _QWORD *v33; // r12
  __int64 *v34; // rax
  __int64 v35; // r12
  unsigned __int8 v36; // al
  __int64 v37; // rax
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 v40; // rax
  int v41; // [rsp+4h] [rbp-16Ch]
  _BYTE *v44; // [rsp+30h] [rbp-140h] BYREF
  __int64 v45; // [rsp+38h] [rbp-138h]
  _BYTE v46[304]; // [rsp+40h] [rbp-130h] BYREF

  v14 = *(_QWORD *)(a3 + 8);
  v44 = v46;
  v45 = 0x2000000000LL;
  if ( !v14 )
    goto LABEL_31;
  while ( 1 )
  {
    *(_QWORD *)&v44[8 * (unsigned int)v45] = v14;
    v15 = v45 + 1;
    LODWORD(v45) = v45 + 1;
    v14 = *(_QWORD *)(v14 + 8);
    if ( !v14 )
      break;
    if ( v15 > 0x1F )
    {
      v29 = v44;
      v30 = 0;
      goto LABEL_26;
    }
    if ( v15 >= HIDWORD(v45) )
      sub_16CD150((__int64)&v44, v46, 0, 8, a5, a6);
  }
  if ( !v15 )
  {
LABEL_31:
    v32 = **a4;
    if ( *((_BYTE *)*a4 + 8) == 16 )
    {
      v33 = (*a4)[4];
      v34 = (__int64 *)sub_1643320(v32);
      v35 = (__int64)sub_16463B0(v34, (unsigned int)v33);
    }
    else
    {
      v35 = sub_1643320(v32);
    }
    v36 = sub_15FF820(*(_WORD *)(a2 + 18) & 0x7FFF);
    v37 = sub_15A0680(v35, v36 ^ 1u, 0);
    v40 = sub_170E100(a1, a2, v37, a7, a8, a9, a10, v38, v39, a13, a14);
    v29 = v44;
    v30 = v40;
    goto LABEL_26;
  }
  v16 = 32;
  LODWORD(v17) = v15;
  v41 = 0;
  while ( 1 )
  {
    v18 = v44;
    v19 = (unsigned int)v17;
    v17 = (unsigned int)(v17 - 1);
    v20 = *(_QWORD **)&v44[8 * v19 - 8];
    LODWORD(v45) = v17;
    v21 = sub_1648700((__int64)v20);
    v24 = *((unsigned __int8 *)v21 + 16);
    if ( (unsigned __int8)v24 <= 0x17u )
    {
LABEL_25:
      v29 = v18;
      v30 = 0;
      goto LABEL_26;
    }
    --v16;
    if ( (unsigned __int8)(v24 - 56) <= 0x17u )
      break;
    if ( (_BYTE)v24 == 54 )
      goto LABEL_30;
    if ( (_BYTE)v24 == 55 )
    {
      if ( *v20 == *(v21 - 6) )
        goto LABEL_25;
      goto LABEL_30;
    }
LABEL_19:
    if ( (_BYTE)v24 == 75 )
    {
      if ( v41 )
        goto LABEL_25;
      v41 = 1;
    }
    else
    {
      if ( (_BYTE)v24 != 78 )
        goto LABEL_25;
      v27 = *(v21 - 3);
      if ( *(_BYTE *)(v27 + 16) )
        goto LABEL_25;
      if ( (*(_BYTE *)(v27 + 33) & 0x20) == 0 )
        goto LABEL_25;
      v28 = *(_DWORD *)(v27 + 36) - 116;
      if ( v28 > 0x15 || ((1LL << v28) & 0x2A0003) == 0 )
        goto LABEL_25;
    }
LABEL_30:
    if ( !(_DWORD)v17 )
      goto LABEL_31;
  }
  v25 = 10518529;
  if ( !_bittest64(&v25, (unsigned int)(v24 - 56)) )
    goto LABEL_19;
  v26 = v21[1];
  if ( !v26 )
    goto LABEL_30;
  while ( v16 > (unsigned int)v17 )
  {
    if ( HIDWORD(v45) <= (unsigned int)v17 )
    {
      sub_16CD150((__int64)&v44, v46, 0, 8, v22, v23);
      v17 = (unsigned int)v45;
    }
    *(_QWORD *)&v44[8 * v17] = v26;
    v17 = (unsigned int)(v45 + 1);
    LODWORD(v45) = v45 + 1;
    v26 = *(_QWORD *)(v26 + 8);
    if ( !v26 )
      goto LABEL_30;
  }
  v30 = 0;
  v29 = v44;
LABEL_26:
  if ( v29 != v46 )
    _libc_free((unsigned __int64)v29);
  return v30;
}
