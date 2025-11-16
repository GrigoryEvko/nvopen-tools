// Function: sub_306CFB0
// Address: 0x306cfb0
//
unsigned __int64 __fastcall sub_306CFB0(__int64 a1, int a2, __int64 a3, int a4, unsigned int a5)
{
  unsigned __int64 v5; // r13
  unsigned int v6; // ebx
  __int64 *v7; // r15
  unsigned int v8; // eax
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r12
  unsigned __int16 i; // r14
  unsigned __int16 v14; // ax
  __int64 v15; // r12
  unsigned int v16; // ebx
  unsigned __int16 v17; // dx
  unsigned int v18; // r13d
  unsigned int v19; // r14d
  __int64 v20; // r13
  __int64 v21; // r12
  signed __int64 v22; // rax
  int v23; // ecx
  int v24; // edx
  signed __int64 v25; // rax
  int v26; // esi
  int v27; // edx
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rax
  unsigned __int64 v31; // rbx
  signed __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 (__fastcall *v41)(__int64, __int64, __int64, __int64, __int64 *); // rax
  unsigned int v42; // eax
  bool v43; // of
  __int64 v44; // rbx
  unsigned __int16 v46; // r9
  bool v47; // cc
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rax
  __int128 v52; // [rsp-18h] [rbp-168h]
  __int128 v53; // [rsp-18h] [rbp-168h]
  __int64 *v54; // [rsp+8h] [rbp-148h]
  unsigned int v55; // [rsp+10h] [rbp-140h]
  int v56; // [rsp+14h] [rbp-13Ch]
  unsigned int v57; // [rsp+30h] [rbp-120h]
  __int64 v59; // [rsp+38h] [rbp-118h]
  unsigned __int64 v60; // [rsp+40h] [rbp-110h]
  unsigned __int64 v62; // [rsp+48h] [rbp-108h]
  unsigned int v64; // [rsp+54h] [rbp-FCh]
  int v65; // [rsp+54h] [rbp-FCh]
  int v66; // [rsp+58h] [rbp-F8h]
  unsigned __int16 v67; // [rsp+58h] [rbp-F8h]
  unsigned __int64 v68; // [rsp+68h] [rbp-E8h]
  __int64 *v69; // [rsp+70h] [rbp-E0h] BYREF
  __int64 *v70; // [rsp+78h] [rbp-D8h]
  _BYTE v71[8]; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v72; // [rsp+88h] [rbp-C8h]
  __int64 v73; // [rsp+90h] [rbp-C0h]
  _BYTE *v74; // [rsp+98h] [rbp-B8h]
  _BYTE v75[32]; // [rsp+A8h] [rbp-A8h] BYREF
  _BYTE *v76; // [rsp+C8h] [rbp-88h]
  _BYTE v77[120]; // [rsp+D8h] [rbp-78h] BYREF

  v5 = 0;
  if ( *(_BYTE *)(a3 + 8) == 18 )
    return v5;
  v6 = *(_DWORD *)(a3 + 32);
  v7 = (__int64 *)a3;
  v55 = -1;
  v54 = *(__int64 **)(a3 + 24);
  if ( v6 )
  {
    _BitScanReverse(&v8, v6);
    v55 = 31 - (v8 ^ 0x1F);
  }
  v64 = *(_DWORD *)(a3 + 32);
  v9 = *(_QWORD *)a3;
  v10 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 0);
  v12 = v11;
  for ( i = v10; ; i = v72 )
  {
    LOWORD(v10) = i;
    sub_2FE6CC0((__int64)v71, *(_QWORD *)(a1 + 24), v9, v10, v12);
    v14 = v72;
    if ( v71[0] == 10 )
      break;
    if ( !v71[0] )
    {
      v46 = i;
      v15 = a1;
      v16 = v64;
      v18 = a5;
      goto LABEL_57;
    }
    if ( i == (_WORD)v72 )
    {
      if ( (_WORD)v72 )
      {
        v15 = a1;
        v16 = v64;
        v17 = v72 - 17;
        v18 = a5;
        goto LABEL_10;
      }
      if ( v12 == v73 )
      {
        v15 = a1;
        v16 = v64;
        v18 = a5;
        goto LABEL_11;
      }
    }
    v10 = v72;
    v12 = v73;
  }
  v46 = i;
  v15 = a1;
  v16 = v64;
  v18 = a5;
  if ( !i )
    goto LABEL_11;
LABEL_57:
  v17 = v46 - 17;
  v14 = v46;
LABEL_10:
  if ( v17 > 0xD3u )
  {
LABEL_11:
    v57 = 1;
    goto LABEL_12;
  }
  if ( (unsigned __int16)(v14 - 176) <= 0x34u )
  {
    v67 = v14;
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
    v14 = v67;
  }
  v57 = word_4456340[v14 - 1];
LABEL_12:
  if ( v57 >= v16 )
  {
    v60 = 0;
    v62 = 0;
  }
  else
  {
    v56 = 0;
    v66 = 0;
    v60 = 0;
    v65 = 0;
    v62 = 0;
    v19 = v18;
    v20 = v15;
    v21 = (__int64)v7;
    do
    {
      v16 >>= 1;
      v59 = v21;
      v21 = sub_BCDA70(v54, v16);
      v22 = sub_30690B0(v20, 5, v59, 0, 0, v19, v16, v21);
      v23 = 1;
      if ( v24 != 1 )
        v23 = v66;
      v66 = v23;
      if ( __OFADD__(v22, v60) )
      {
        v47 = v22 <= 0;
        v48 = 0x8000000000000000LL;
        if ( !v47 )
          v48 = 0x7FFFFFFFFFFFFFFFLL;
        v60 = v48;
      }
      else
      {
        v60 += v22;
      }
      v69 = (__int64 *)v21;
      v70 = (__int64 *)v21;
      v68 = v68 & 0xFFFFFFFF00000000LL | 1;
      *((_QWORD *)&v52 + 1) = v68;
      *(_QWORD *)&v52 = 0;
      sub_DF8CB0((__int64)v71, a2, v21, (char *)&v69, 2, a4, 0, v52);
      v25 = sub_306A930(v20, (__int64)v71, v19);
      v26 = 1;
      if ( v27 != 1 )
        v26 = v65;
      v65 = v26;
      if ( __OFADD__(v25, v62) )
      {
        v47 = v25 <= 0;
        v49 = 0x8000000000000000LL;
        if ( !v47 )
          v49 = 0x7FFFFFFFFFFFFFFFLL;
        v62 = v49;
      }
      else
      {
        v62 += v25;
      }
      ++v56;
      if ( v76 != v77 )
        _libc_free((unsigned __int64)v76);
      if ( v74 != v75 )
        _libc_free((unsigned __int64)v74);
    }
    while ( v57 < v16 );
    v55 -= v56;
    v7 = (__int64 *)v21;
    v15 = v20;
    v18 = v19;
  }
  v28 = sub_30690B0(v15, 7, (__int64)v7, 0, 0, v18, 0, (__int64)v7);
  if ( v29 == 1 )
  {
    if ( is_mul_ok(v28, v55) )
    {
      v30 = v28 * v55;
      goto LABEL_31;
    }
    if ( v28 > 0 && v55 )
    {
      v51 = 0x7FFFFFFFFFFFFFFFLL;
      v31 = v60 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v60) )
        goto LABEL_32;
    }
    else
    {
      v51 = 0x8000000000000000LL;
      v31 = v60 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v60) )
        goto LABEL_32;
    }
LABEL_80:
    v31 = v51;
    goto LABEL_32;
  }
  if ( !is_mul_ok(v28, v55) )
  {
    if ( v55 && v28 > 0 )
    {
      v51 = 0x7FFFFFFFFFFFFFFFLL;
      v31 = v60 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v60) )
        goto LABEL_32;
    }
    else
    {
      v51 = 0x8000000000000000LL;
      v31 = v60 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v60) )
        goto LABEL_32;
    }
    goto LABEL_80;
  }
  v30 = v28 * v55;
LABEL_31:
  v31 = v30 + v60;
  if ( __OFADD__(v30, v60) )
  {
    v31 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v30 <= 0 )
      v31 = 0x8000000000000000LL;
  }
LABEL_32:
  v69 = v7;
  v70 = v7;
  *((_QWORD *)&v53 + 1) = 1;
  *(_QWORD *)&v53 = 0;
  sub_DF8CB0((__int64)v71, a2, (__int64)v7, (char *)&v69, 2, a4, 0, v53);
  v32 = sub_306A930(v15, (__int64)v71, v18);
  if ( v33 == 1 )
  {
    if ( is_mul_ok(v32, v55) )
    {
      v34 = v32 * v55;
      goto LABEL_35;
    }
    if ( v55 && v32 > 0 )
    {
      v50 = 0x7FFFFFFFFFFFFFFFLL;
      v35 = v62 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v62) )
        goto LABEL_36;
    }
    else
    {
      v50 = 0x8000000000000000LL;
      v35 = v62 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v62) )
        goto LABEL_36;
    }
LABEL_74:
    v35 = v50;
    goto LABEL_36;
  }
  if ( !is_mul_ok(v32, v55) )
  {
    if ( v55 && v32 > 0 )
    {
      v50 = 0x7FFFFFFFFFFFFFFFLL;
      v35 = v62 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v62) )
        goto LABEL_36;
    }
    else
    {
      v50 = 0x8000000000000000LL;
      v35 = v62 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v62) )
        goto LABEL_36;
    }
    goto LABEL_74;
  }
  v34 = v32 * v55;
LABEL_35:
  v35 = v34 + v62;
  if ( __OFADD__(v34, v62) )
  {
    v35 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v34 <= 0 )
      v35 = 0x8000000000000000LL;
  }
LABEL_36:
  if ( (unsigned int)*((unsigned __int8 *)v7 + 8) - 17 <= 1 )
    v7 = *(__int64 **)v7[2];
  v36 = sub_2D5BAE0(*(_QWORD *)(v15 + 24), *(_QWORD *)(v15 + 8), v7, 0);
  v37 = *(_QWORD *)(v15 + 24);
  v39 = v38;
  v40 = v36;
  v41 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64 *))(*(_QWORD *)v37 + 736LL);
  BYTE2(v69) = 0;
  v42 = v41(v37, *v7, v40, v39, v69);
  v43 = __OFADD__(v35, v31);
  v44 = v35 + v31;
  if ( !v43 )
    goto LABEL_39;
  if ( v35 > 0 )
  {
    v44 = 0x7FFFFFFFFFFFFFFFLL;
LABEL_39:
    v5 = v42 + v44;
    if ( __OFADD__(v42, v44) )
    {
      v5 = 0x7FFFFFFFFFFFFFFFLL;
      if ( !v42 )
        v5 = 0x8000000000000000LL;
    }
    goto LABEL_40;
  }
  v5 = v42 + 0x8000000000000000LL;
LABEL_40:
  if ( v76 != v77 )
    _libc_free((unsigned __int64)v76);
  if ( v74 != v75 )
    _libc_free((unsigned __int64)v74);
  return v5;
}
