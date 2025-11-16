// Function: sub_13F9660
// Address: 0x13f9660
//
__int64 __fastcall sub_13F9660(__int64 *a1, _QWORD *a2, _BYTE *a3, int a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r14
  unsigned __int16 v9; // ax
  __int64 v10; // r13
  unsigned __int64 v11; // r13
  _QWORD *v12; // r12
  __int64 v13; // r13
  __int64 v14; // r14
  char v15; // al
  _QWORD *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v25; // rax
  int v26; // eax
  __int64 *v27; // r13
  __int64 *v28; // r12
  __int64 v29; // rcx
  __int64 *v30; // rbx
  __int64 v31; // rsi
  __m128i v32; // xmm0
  __m128i v33; // xmm1
  __int64 v36; // [rsp+30h] [rbp-100h]
  int v37; // [rsp+30h] [rbp-100h]
  __int64 v38; // [rsp+38h] [rbp-F8h]
  unsigned __int8 v39; // [rsp+47h] [rbp-E9h]
  _QWORD *v40; // [rsp+48h] [rbp-E8h]
  __m128i v41; // [rsp+50h] [rbp-E0h] BYREF
  __m128i v42; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v43; // [rsp+70h] [rbp-C0h]
  __m128i v44[2]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v45; // [rsp+A0h] [rbp-90h]
  char v46; // [rsp+A8h] [rbp-88h]
  __int64 *v47; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v48; // [rsp+B8h] [rbp-78h]
  _BYTE v49[112]; // [rsp+C0h] [rbp-70h] BYREF

  v5 = sub_15F2050(a1);
  v36 = sub_1632FA0(v5);
  v6 = sub_1649C60(*(a1 - 3));
  v7 = a1[5];
  v8 = *a1;
  v38 = v6;
  v39 = sub_15F32D0(a1);
  v9 = *((_WORD *)a1 + 9);
  if ( ((v9 >> 7) & 6) != 0 )
    return 0;
  if ( (v9 & 1) != 0 )
    return 0;
  v10 = a1[3];
  v47 = (__int64 *)v49;
  v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
  v48 = 0x800000000LL;
  v40 = (_QWORD *)(v7 + 40);
  if ( v11 == v7 + 40 )
    return 0;
  v12 = (_QWORD *)v11;
  v13 = v8;
  v14 = v36;
  while ( 1 )
  {
    if ( !v12 )
      BUG();
    v15 = *((_BYTE *)v12 - 8);
    v16 = v12 - 3;
    if ( v15 != 78 )
      break;
    v25 = *(v12 - 6);
    if ( *(_BYTE *)(v25 + 16) || (*(_BYTE *)(v25 + 33) & 0x20) == 0 )
    {
      v26 = a4 - 1;
      if ( !a4 )
        goto LABEL_35;
      goto LABEL_32;
    }
    if ( (unsigned int)(*(_DWORD *)(v25 + 36) - 35) > 3 )
    {
      v26 = a4 - 1;
      if ( !a4 )
        goto LABEL_35;
LABEL_32:
      a4 = v26;
LABEL_23:
      if ( (unsigned __int8)sub_15F3040(v12 - 3) )
      {
        v23 = (unsigned int)v48;
        if ( (unsigned int)v48 >= HIDWORD(v48) )
        {
          sub_16CD150(&v47, v49, 0, 8);
          v23 = (unsigned int)v48;
        }
        v47[v23] = (__int64)v16;
        LODWORD(v48) = v48 + 1;
      }
    }
    v12 = (_QWORD *)(*v12 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v40 == v12 )
      goto LABEL_35;
  }
  v37 = a4 - 1;
  if ( !a4 )
    goto LABEL_35;
  if ( v15 != 54 )
    goto LABEL_14;
  if ( v39 > (unsigned __int8)sub_15F32D0(v12 - 3)
    || (v17 = sub_1649C60(*(v12 - 6)), !(unsigned __int8)sub_13F74E0(v17, v38)) )
  {
LABEL_22:
    a4 = v37;
    goto LABEL_23;
  }
  if ( !(unsigned __int8)sub_15FBDD0(*(v12 - 3), v13, v14) )
  {
    v15 = *((_BYTE *)v12 - 8);
LABEL_14:
    if ( v15 == 55 && v39 <= (unsigned __int8)sub_15F32D0(v12 - 3) )
    {
      v21 = sub_1649C60(*(v12 - 6));
      if ( (unsigned __int8)sub_13F74E0(v21, v38) )
      {
        if ( a3 )
          *a3 = 0;
        v22 = *(v12 - 9);
        if ( (unsigned __int8)sub_15FBDD0(*(_QWORD *)v22, v13, v14)
          || *(_BYTE *)(v22 + 16) <= 0x10u && (v22 = sub_14D66F0(v22, v13, v14)) != 0 )
        {
          v28 = a1;
          goto LABEL_41;
        }
      }
    }
    goto LABEL_22;
  }
  v28 = a1;
  v22 = (__int64)v16;
  if ( a3 )
    *a3 = 1;
LABEL_41:
  sub_141EB40(&v41, v28, v18, v19, v20);
  v30 = v47;
  v27 = &v47[(unsigned int)v48];
  if ( v47 != v27 )
  {
    while ( 1 )
    {
      v31 = *v30;
      v32 = _mm_loadu_si128(&v41);
      v33 = _mm_loadu_si128(&v42);
      v46 = 1;
      v45 = v43;
      v44[0] = v32;
      v44[1] = v33;
      if ( (sub_13575E0(a2, v31, v44, v29) & 2) != 0 )
        break;
      if ( v27 == ++v30 )
      {
        v27 = v47;
        goto LABEL_36;
      }
    }
LABEL_35:
    v27 = v47;
    v22 = 0;
  }
LABEL_36:
  if ( v27 != (__int64 *)v49 )
    _libc_free((unsigned __int64)v27);
  return v22;
}
