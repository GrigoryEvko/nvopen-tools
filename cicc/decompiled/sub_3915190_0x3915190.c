// Function: sub_3915190
// Address: 0x3915190
//
__int64 __fastcall sub_3915190(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v3; // r13
  __int64 v5; // rbx
  __int64 v6; // r14
  _QWORD *v7; // rax
  unsigned __int64 v8; // rax
  char v9; // cl
  char v10; // dl
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int32 v14; // esi
  __int64 v15; // rdi
  char *v16; // rax
  __int64 v17; // rdi
  char *v18; // rax
  __int16 v19; // si
  char v20; // al
  __int64 v21; // rdi
  __int16 v22; // dx
  __int64 v23; // rdi
  unsigned __int64 v24; // rdx
  unsigned int v26; // ecx
  unsigned int v27; // ebx
  char v28; // cl
  unsigned int v29; // edx
  unsigned int v30; // eax
  unsigned __int32 v31; // eax
  char v32; // al
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // rax
  unsigned int v45; // eax
  __int64 v46; // rdi
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  char v52; // [rsp+0h] [rbp-170h]
  char v53; // [rsp+Eh] [rbp-162h]
  char v54; // [rsp+Eh] [rbp-162h]
  char v55; // [rsp+Eh] [rbp-162h]
  char v56; // [rsp+Eh] [rbp-162h]
  char v57; // [rsp+Fh] [rbp-161h]
  _QWORD v58[2]; // [rsp+10h] [rbp-160h] BYREF
  __m128i v59; // [rsp+20h] [rbp-150h] BYREF
  char v60; // [rsp+30h] [rbp-140h]
  char v61; // [rsp+31h] [rbp-13Fh]
  __m128i v62; // [rsp+40h] [rbp-130h] BYREF
  __int16 v63; // [rsp+50h] [rbp-120h]
  __m128i v64[2]; // [rsp+60h] [rbp-110h] BYREF
  __m128i v65; // [rsp+80h] [rbp-F0h] BYREF
  char v66; // [rsp+90h] [rbp-E0h]
  char v67; // [rsp+91h] [rbp-DFh]
  __m128i v68[2]; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i v69; // [rsp+C0h] [rbp-B0h] BYREF
  __int16 v70; // [rsp+D0h] [rbp-A0h]
  __m128i v71[2]; // [rsp+E0h] [rbp-90h] BYREF
  __m128i v72; // [rsp+100h] [rbp-70h] BYREF
  char v73; // [rsp+110h] [rbp-60h]
  char v74; // [rsp+111h] [rbp-5Fh]
  __m128i v75[5]; // [rsp+120h] [rbp-50h] BYREF

  v5 = *(_QWORD *)a2;
  v6 = sub_3914E30(a1, *(_QWORD *)a2);
  v57 = *(_BYTE *)(a2 + 16);
  if ( v5 == v6 )
  {
    v33 = *(_QWORD *)v5;
  }
  else
  {
    v7 = sub_3914DB0(a1, v6);
    v3 = v7;
    if ( v7 )
      v57 = *((_BYTE *)v7 + 16);
    v8 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 )
      goto LABEL_59;
    v9 = 10;
    if ( (*(_BYTE *)(v6 + 9) & 0xC) != 8 )
      goto LABEL_6;
    *(_BYTE *)(v6 + 8) |= 4u;
    v40 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v6 + 24));
    v9 = 10;
    v41 = v40;
    v33 = v40 | *(_QWORD *)v6 & 7LL;
    *(_QWORD *)v6 = v33;
    if ( !v41 )
    {
LABEL_56:
      v8 = v33 & 0xFFFFFFFFFFFFFFF8LL;
      goto LABEL_6;
    }
  }
  v8 = v33 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v8 )
  {
    v9 = 0;
    if ( (*(_BYTE *)(v6 + 9) & 0xC) != 8 )
      goto LABEL_6;
    *(_BYTE *)(v6 + 8) |= 4u;
    v34 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v6 + 24));
    v9 = 0;
    v35 = v34;
    v33 = v34 | *(_QWORD *)v6 & 7LL;
    *(_QWORD *)v6 = v33;
    if ( !v35 )
      goto LABEL_56;
    v8 = v33 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v8 )
    {
      v36 = 0;
      if ( (*(_BYTE *)(v6 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v6 + 8) |= 4u;
        v36 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v6 + 24));
        v37 = v36 | *(_QWORD *)v6 & 7LL;
        *(_QWORD *)v6 = v37;
        v8 = v37 & 0xFFFFFFFFFFFFFFF8LL;
      }
      goto LABEL_60;
    }
  }
LABEL_59:
  v36 = v8;
LABEL_60:
  v9 = 2;
  if ( off_4CF6DB8 != (_UNKNOWN *)v36 )
    v9 = 14;
LABEL_6:
  v10 = *(_BYTE *)(v5 + 8);
  if ( (v10 & 0x20) != 0 )
    v9 |= 0x10u;
  if ( (v10 & 0x10) != 0 )
  {
    v9 |= 1u;
    if ( v5 != v6 )
      goto LABEL_10;
  }
  else
  {
    if ( v5 != v6 )
    {
LABEL_10:
      if ( v8 )
        goto LABEL_52;
      if ( (*(_BYTE *)(v6 + 9) & 0xC) != 8
        || (*(_BYTE *)(v6 + 8) |= 4u,
            v55 = v9,
            v42 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v6 + 24)),
            v9 = v55,
            v43 = v42,
            v44 = v42 | *(_QWORD *)v6 & 7LL,
            *(_QWORD *)v6 = v44,
            !v43) )
      {
        v11 = v3[1];
        goto LABEL_13;
      }
      v8 = v44 & 0xFFFFFFFFFFFFFFF8LL;
      goto LABEL_40;
    }
    if ( v8 )
      goto LABEL_52;
    if ( (*(_BYTE *)(v5 + 9) & 0xC) != 8 )
    {
      v9 |= 1u;
      goto LABEL_41;
    }
    v46 = *(_QWORD *)(v5 + 24);
    v56 = v9;
    *(_BYTE *)(v5 + 8) = v10 | 4;
    v47 = (unsigned __int64)sub_38CE440(v46);
    v9 = v56;
    v48 = v47;
    v49 = v47 | *(_QWORD *)v5 & 7LL;
    *(_QWORD *)v5 = v49;
    v8 = v49 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v48 )
      v9 = v56 | 1;
  }
LABEL_40:
  if ( v8 )
    goto LABEL_52;
LABEL_41:
  v32 = *(_BYTE *)(v6 + 9) & 0xC;
  if ( v32 != 8 )
    goto LABEL_42;
  *(_BYTE *)(v6 + 8) |= 4u;
  v53 = v9;
  v38 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v6 + 24));
  v9 = v53;
  *(_QWORD *)v6 = v38 | *(_QWORD *)v6 & 7LL;
  if ( v38 )
  {
LABEL_52:
    v54 = v9;
    v39 = sub_3913FA0(a1, v5, a3);
    v9 = v54;
    v11 = v39;
    goto LABEL_13;
  }
  v32 = *(_BYTE *)(v6 + 9) & 0xC;
LABEL_42:
  v11 = 0;
  if ( v32 == 12 )
    v11 = *(_QWORD *)(v6 + 24);
LABEL_13:
  v52 = v9;
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(_QWORD *)(a1 + 240);
  v14 = _byteswap_ulong(v12);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    LODWORD(v12) = v14;
  v75[0].m128i_i32[0] = v12;
  sub_16E7EE0(v13, v75[0].m128i_i8, 4u);
  v15 = *(_QWORD *)(a1 + 240);
  v16 = *(char **)(v15 + 24);
  if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 16) )
  {
    sub_16E7DE0(v15, v52);
  }
  else
  {
    *(_QWORD *)(v15 + 24) = v16 + 1;
    *v16 = v52;
  }
  v17 = *(_QWORD *)(a1 + 240);
  v18 = *(char **)(v17 + 24);
  if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 16) )
  {
    sub_16E7DE0(v17, v57);
  }
  else
  {
    *(_QWORD *)(v17 + 24) = v18 + 1;
    *v18 = v57;
  }
  v19 = *(_WORD *)(v6 + 12);
  v20 = *(_BYTE *)(v6 + 9) & 0xC;
  if ( v5 == v6 || (*(_BYTE *)(v5 + 13) & 2) == 0 )
  {
    if ( v20 != 12 )
      goto LABEL_23;
    v26 = *(_DWORD *)(v6 + 8);
    if ( (v26 & 0x1F000) == 0 )
      goto LABEL_23;
    v27 = 1 << (((v26 >> 12) & 0x1F) - 1);
    if ( !v27 )
      goto LABEL_23;
    v28 = 0;
  }
  else
  {
    if ( v20 != 12 || (v45 = *(_DWORD *)(v6 + 8), (v45 & 0x1F000) == 0) || (v27 = 1 << (((v45 >> 12) & 0x1F) - 1)) == 0 )
    {
LABEL_22:
      v19 |= 0x200u;
      goto LABEL_23;
    }
    v28 = 1;
  }
  _BitScanReverse(&v29, v27);
  v30 = 31 - (v29 ^ 0x1F);
  if ( v30 > 0xF )
  {
    v74 = 1;
    v72.m128i_i64[0] = (__int64)"'";
    v73 = 3;
    v70 = 261;
    v58[0] = sub_3913870((_BYTE *)v6);
    v69.m128i_i64[0] = (__int64)v58;
    v58[1] = v50;
    v65.m128i_i64[0] = (__int64)"' for '";
    v59.m128i_i64[0] = (__int64)"invalid 'common' alignment '";
    v67 = 1;
    v66 = 3;
    v63 = 265;
    v62.m128i_i32[0] = v27;
    v61 = 1;
    v60 = 3;
    sub_14EC200(v64, &v59, &v62);
    sub_14EC200(v68, v64, &v65);
    sub_14EC200(v71, v68, &v69);
    sub_14EC200(v75, v71, &v72);
    sub_16BCFB0((__int64)v75, 0);
  }
  v19 = ((_WORD)v30 << 8) | v19 & 0xF0FF;
  if ( v28 )
    goto LABEL_22;
LABEL_23:
  v21 = *(_QWORD *)(a1 + 240);
  v22 = __ROL2__(v19, 8);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v19 = v22;
  v75[0].m128i_i16[0] = v19;
  sub_16E7EE0(v21, v75[0].m128i_i8, 2u);
  v23 = *(_QWORD *)(a1 + 240);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) & 1) != 0 )
  {
    v24 = _byteswap_uint64(v11);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
      v11 = v24;
    v75[0].m128i_i64[0] = v11;
    return sub_16E7EE0(v23, v75[0].m128i_i8, 8u);
  }
  else
  {
    v31 = _byteswap_ulong(v11);
    if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
      LODWORD(v11) = v31;
    v75[0].m128i_i32[0] = v11;
    return sub_16E7EE0(v23, v75[0].m128i_i8, 4u);
  }
}
