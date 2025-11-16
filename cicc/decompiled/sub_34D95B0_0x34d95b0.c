// Function: sub_34D95B0
// Address: 0x34d95b0
//
unsigned __int64 __fastcall sub_34D95B0(__int64 a1, int a2, __int64 a3, int a4, unsigned int a5)
{
  unsigned __int64 v5; // r13
  unsigned int v6; // ebx
  __int64 v7; // r15
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
  __int64 v25; // rax
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
  bool v37; // of
  __int64 v38; // rbx
  unsigned __int16 v40; // r9
  bool v41; // cc
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rax
  __int128 v47; // [rsp-18h] [rbp-168h]
  __int128 v48; // [rsp-18h] [rbp-168h]
  __int64 *v49; // [rsp+8h] [rbp-148h]
  unsigned int v50; // [rsp+10h] [rbp-140h]
  int v51; // [rsp+14h] [rbp-13Ch]
  unsigned int v52; // [rsp+30h] [rbp-120h]
  __int64 v54; // [rsp+38h] [rbp-118h]
  unsigned __int64 v55; // [rsp+40h] [rbp-110h]
  unsigned __int64 v57; // [rsp+48h] [rbp-108h]
  unsigned int v59; // [rsp+54h] [rbp-FCh]
  int v60; // [rsp+54h] [rbp-FCh]
  int v61; // [rsp+58h] [rbp-F8h]
  unsigned __int16 v62; // [rsp+58h] [rbp-F8h]
  unsigned __int64 v63; // [rsp+68h] [rbp-E8h]
  __int64 v64; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v65; // [rsp+78h] [rbp-D8h]
  _BYTE v66[8]; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v67; // [rsp+88h] [rbp-C8h]
  __int64 v68; // [rsp+90h] [rbp-C0h]
  _BYTE *v69; // [rsp+98h] [rbp-B8h]
  _BYTE v70[32]; // [rsp+A8h] [rbp-A8h] BYREF
  _BYTE *v71; // [rsp+C8h] [rbp-88h]
  _BYTE v72[120]; // [rsp+D8h] [rbp-78h] BYREF

  v5 = 0;
  if ( *(_BYTE *)(a3 + 8) == 18 )
    return v5;
  v6 = *(_DWORD *)(a3 + 32);
  v7 = a3;
  v50 = -1;
  v49 = *(__int64 **)(a3 + 24);
  if ( v6 )
  {
    _BitScanReverse(&v8, v6);
    v50 = 31 - (v8 ^ 0x1F);
  }
  v59 = *(_DWORD *)(a3 + 32);
  v9 = *(_QWORD *)a3;
  v10 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 0);
  v12 = v11;
  for ( i = v10; ; i = v67 )
  {
    LOWORD(v10) = i;
    sub_2FE6CC0((__int64)v66, *(_QWORD *)(a1 + 24), v9, v10, v12);
    v14 = v67;
    if ( v66[0] == 10 )
      break;
    if ( !v66[0] )
    {
      v40 = i;
      v15 = a1;
      v16 = v59;
      v18 = a5;
      goto LABEL_57;
    }
    if ( i == (_WORD)v67 )
    {
      if ( (_WORD)v67 )
      {
        v15 = a1;
        v16 = v59;
        v17 = v67 - 17;
        v18 = a5;
        goto LABEL_10;
      }
      if ( v12 == v68 )
      {
        v15 = a1;
        v16 = v59;
        v18 = a5;
        goto LABEL_11;
      }
    }
    v10 = v67;
    v12 = v68;
  }
  v40 = i;
  v15 = a1;
  v16 = v59;
  v18 = a5;
  if ( !i )
    goto LABEL_11;
LABEL_57:
  v17 = v40 - 17;
  v14 = v40;
LABEL_10:
  if ( v17 > 0xD3u )
  {
LABEL_11:
    v52 = 1;
    goto LABEL_12;
  }
  if ( (unsigned __int16)(v14 - 176) <= 0x34u )
  {
    v62 = v14;
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
    v14 = v62;
  }
  v52 = word_4456340[v14 - 1];
LABEL_12:
  if ( v52 >= v16 )
  {
    v55 = 0;
    v57 = 0;
  }
  else
  {
    v51 = 0;
    v61 = 0;
    v55 = 0;
    v60 = 0;
    v57 = 0;
    v19 = v18;
    v20 = v15;
    v21 = v7;
    do
    {
      v16 >>= 1;
      v54 = v21;
      v21 = sub_BCDA70(v49, v16);
      v22 = sub_34D5BE0(v20, 5, v54, 0, 0, v19, v16, v21);
      v23 = 1;
      if ( v24 != 1 )
        v23 = v61;
      v61 = v23;
      if ( __OFADD__(v22, v55) )
      {
        v41 = v22 <= 0;
        v42 = 0x8000000000000000LL;
        if ( !v41 )
          v42 = 0x7FFFFFFFFFFFFFFFLL;
        v55 = v42;
      }
      else
      {
        v55 += v22;
      }
      v64 = v21;
      v65 = v21;
      v63 = v63 & 0xFFFFFFFF00000000LL | 1;
      *((_QWORD *)&v47 + 1) = v63;
      *(_QWORD *)&v47 = 0;
      sub_DF8CB0((__int64)v66, a2, v21, (char *)&v64, 2, a4, 0, v47);
      v25 = sub_34D6FB0(v20, (__int64)v66, v19);
      v26 = 1;
      if ( v27 != 1 )
        v26 = v60;
      v60 = v26;
      if ( __OFADD__(v25, v57) )
      {
        v41 = v25 <= 0;
        v43 = 0x8000000000000000LL;
        if ( !v41 )
          v43 = 0x7FFFFFFFFFFFFFFFLL;
        v57 = v43;
      }
      else
      {
        v57 += v25;
      }
      ++v51;
      if ( v71 != v72 )
        _libc_free((unsigned __int64)v71);
      if ( v69 != v70 )
        _libc_free((unsigned __int64)v69);
    }
    while ( v52 < v16 );
    v50 -= v51;
    v7 = v21;
    v15 = v20;
    v18 = v19;
  }
  v28 = sub_34D5BE0(v15, 7, v7, 0, 0, v18, 0, v7);
  if ( v29 == 1 )
  {
    if ( is_mul_ok(v28, v50) )
    {
      v30 = v28 * v50;
      goto LABEL_31;
    }
    if ( v50 && v28 > 0 )
    {
      v45 = 0x7FFFFFFFFFFFFFFFLL;
      v31 = v55 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v55) )
        goto LABEL_32;
    }
    else
    {
      v45 = 0x8000000000000000LL;
      v31 = v55 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v55) )
        goto LABEL_32;
    }
LABEL_81:
    v31 = v45;
    goto LABEL_32;
  }
  if ( !is_mul_ok(v28, v50) )
  {
    if ( v28 > 0 && v50 )
    {
      v45 = 0x7FFFFFFFFFFFFFFFLL;
      v31 = v55 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v55) )
        goto LABEL_32;
    }
    else
    {
      v45 = 0x8000000000000000LL;
      v31 = v55 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v55) )
        goto LABEL_32;
    }
    goto LABEL_81;
  }
  v30 = v28 * v50;
LABEL_31:
  v31 = v30 + v55;
  if ( __OFADD__(v30, v55) )
  {
    v31 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v30 <= 0 )
      v31 = 0x8000000000000000LL;
  }
LABEL_32:
  v64 = v7;
  v65 = v7;
  *((_QWORD *)&v48 + 1) = 1;
  *(_QWORD *)&v48 = 0;
  sub_DF8CB0((__int64)v66, a2, v7, (char *)&v64, 2, a4, 0, v48);
  v32 = sub_34D6FB0(v15, (__int64)v66, v18);
  if ( v33 != 1 )
  {
    if ( !is_mul_ok(v32, v50) )
    {
      if ( v50 && v32 > 0 )
      {
        v46 = 0x7FFFFFFFFFFFFFFFLL;
        v35 = v57 + 0x7FFFFFFFFFFFFFFFLL;
        if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v57) )
          goto LABEL_36;
      }
      else
      {
        v46 = 0x8000000000000000LL;
        v35 = v57 + 0x8000000000000000LL;
        if ( !__OFADD__(0x8000000000000000LL, v57) )
          goto LABEL_36;
      }
      v35 = v46;
      goto LABEL_36;
    }
    v34 = v32 * v50;
LABEL_35:
    v35 = v34 + v57;
    if ( __OFADD__(v34, v57) )
    {
      v35 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v34 <= 0 )
        v35 = 0x8000000000000000LL;
    }
LABEL_36:
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
      goto LABEL_38;
    goto LABEL_37;
  }
  if ( is_mul_ok(v32, v50) )
  {
    v34 = v32 * v50;
    goto LABEL_35;
  }
  if ( v50 && v32 > 0 )
  {
    v44 = 0x7FFFFFFFFFFFFFFFLL;
    v35 = v57 + 0x7FFFFFFFFFFFFFFFLL;
    if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v57) )
      goto LABEL_74;
  }
  else
  {
    v44 = 0x8000000000000000LL;
    v35 = v57 + 0x8000000000000000LL;
    if ( !__OFADD__(0x8000000000000000LL, v57) )
      goto LABEL_74;
  }
  v35 = v44;
LABEL_74:
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
LABEL_37:
    v7 = **(_QWORD **)(v7 + 16);
LABEL_38:
  v36 = (unsigned int)sub_34D06B0(v15, (__int64 *)v7);
  v37 = __OFADD__(v35, v31);
  v38 = v35 + v31;
  if ( !v37 )
  {
LABEL_39:
    v5 = v36 + v38;
    if ( __OFADD__(v36, v38) )
    {
      v5 = 0x7FFFFFFFFFFFFFFFLL;
      if ( !v36 )
        v5 = 0x8000000000000000LL;
    }
    goto LABEL_40;
  }
  if ( v35 > 0 )
  {
    v38 = 0x7FFFFFFFFFFFFFFFLL;
    goto LABEL_39;
  }
  v5 = v36 + 0x8000000000000000LL;
LABEL_40:
  if ( v71 != v72 )
    _libc_free((unsigned __int64)v71);
  if ( v69 != v70 )
    _libc_free((unsigned __int64)v69);
  return v5;
}
