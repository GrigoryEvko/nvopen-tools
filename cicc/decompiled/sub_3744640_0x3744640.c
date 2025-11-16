// Function: sub_3744640
// Address: 0x3744640
//
__int64 __fastcall sub_3744640(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rax
  int v4; // edx
  __int64 **v5; // rax
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // edx
  __int64 *v15; // r15
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  int v20; // r14d
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int8 *i; // r13
  __int64 v24; // rbx
  __int64 v25; // rax
  __m128i *v26; // rsi
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // rdx
  char v30; // cl
  char v31; // al
  __int64 v32; // rdx
  char v33; // al
  __int64 v34; // rdx
  unsigned __int64 v35; // rdi
  const __m128i *v36; // rax
  __m128i *v37; // rax
  const __m128i *v38; // rax
  int v39; // eax
  unsigned int v40; // r12d
  char v42; // al
  char v43; // cl
  __int64 **v44; // [rsp+20h] [rbp-410h]
  char v46; // [rsp+30h] [rbp-400h]
  __int64 v47; // [rsp+38h] [rbp-3F8h]
  __int64 v48; // [rsp+48h] [rbp-3E8h] BYREF
  const __m128i *v49; // [rsp+50h] [rbp-3E0h] BYREF
  __m128i *v50; // [rsp+58h] [rbp-3D8h]
  const __m128i *v51; // [rsp+60h] [rbp-3D0h]
  __m128i v52; // [rsp+70h] [rbp-3C0h] BYREF
  __m128i v53; // [rsp+80h] [rbp-3B0h] BYREF
  __m128i v54; // [rsp+90h] [rbp-3A0h] BYREF
  __int64 **v55; // [rsp+A0h] [rbp-390h] BYREF
  unsigned __int64 v56; // [rsp+A8h] [rbp-388h]
  __int64 v57; // [rsp+B0h] [rbp-380h]
  __int64 v58; // [rsp+B8h] [rbp-378h]
  __int64 v59; // [rsp+C0h] [rbp-370h]
  const __m128i *v60; // [rsp+C8h] [rbp-368h]
  __m128i *v61; // [rsp+D0h] [rbp-360h]
  const __m128i *v62; // [rsp+D8h] [rbp-358h]
  unsigned __int8 *v63; // [rsp+E0h] [rbp-350h]
  __int64 v64; // [rsp+E8h] [rbp-348h]
  __int64 v65; // [rsp+F0h] [rbp-340h]
  _BYTE *v66; // [rsp+F8h] [rbp-338h]
  __int64 v67; // [rsp+100h] [rbp-330h]
  _BYTE v68[128]; // [rsp+108h] [rbp-328h] BYREF
  _BYTE *v69; // [rsp+188h] [rbp-2A8h]
  __int64 v70; // [rsp+190h] [rbp-2A0h]
  _BYTE v71[256]; // [rsp+198h] [rbp-298h] BYREF
  _BYTE *v72; // [rsp+298h] [rbp-198h]
  __int64 v73; // [rsp+2A0h] [rbp-190h]
  _BYTE v74[64]; // [rsp+2A8h] [rbp-188h] BYREF
  _BYTE *v75; // [rsp+2E8h] [rbp-148h]
  __int64 v76; // [rsp+2F0h] [rbp-140h]
  _BYTE v77[224]; // [rsp+2F8h] [rbp-138h] BYREF
  _BYTE *v78; // [rsp+3D8h] [rbp-58h]
  __int64 v79; // [rsp+3E0h] [rbp-50h]
  _BYTE v80[72]; // [rsp+3E8h] [rbp-48h] BYREF

  v3 = *((_QWORD *)a2 + 10);
  v4 = *a2;
  v49 = 0;
  v47 = v3;
  v5 = (__int64 **)*((_QWORD *)a2 + 1);
  v50 = 0;
  v44 = v5;
  v51 = 0;
  v52 = 0u;
  v53 = 0u;
  v54 = 0u;
  if ( v4 == 40 )
  {
    v6 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v6 = 0;
    if ( v4 != 85 )
    {
      v6 = 64;
      if ( v4 != 34 )
LABEL_78:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v7 = sub_BD2BC0((__int64)a2);
  v9 = v7 + v8;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v9 >> 4) )
LABEL_79:
      BUG();
LABEL_10:
    v13 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v9 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_79;
  v10 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v11 = sub_BD2BC0((__int64)a2);
  v13 = 32LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
LABEL_11:
  sub_3375F60(&v49, (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v6 - v13) >> 5));
  v14 = *a2;
  v15 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( v14 == 40 )
  {
    v16 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v16 = -32;
    if ( v14 != 85 )
    {
      v16 = -96;
      if ( v14 != 34 )
        goto LABEL_78;
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v17 = sub_BD2BC0((__int64)a2);
    v19 = v17 + v18;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v19 >> 4) )
        goto LABEL_24;
    }
    else
    {
      if ( !(unsigned int)((v19 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_24;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v20 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v21 = sub_BD2BC0((__int64)a2);
        v16 -= 32LL * (unsigned int)(*(_DWORD *)(v21 + v22 - 4) - v20);
        goto LABEL_24;
      }
    }
    BUG();
  }
LABEL_24:
  for ( i = &a2[v16]; v15 != (__int64 *)i; v15 += 4 )
  {
    v24 = *v15;
    if ( !(unsigned __int8)sub_BCADB0(*(_QWORD *)(*v15 + 8)) )
    {
      v25 = *(_QWORD *)(v24 + 8);
      v52.m128i_i64[0] = v24;
      v53.m128i_i64[1] = v25;
      sub_34470B0((__int64)&v52, (__int64)a2, ((char *)v15 - (char *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]) >> 5);
      v26 = v50;
      if ( v50 == v51 )
      {
        sub_332CDC0((unsigned __int64 *)&v49, v50, &v52);
      }
      else
      {
        if ( v50 )
        {
          *v50 = _mm_loadu_si128(&v52);
          v26[1] = _mm_loadu_si128(&v53);
          v26[2] = _mm_loadu_si128(&v54);
          v26 = v50;
        }
        v50 = v26 + 3;
      }
    }
  }
  if ( (*((_WORD *)a2 + 1) & 3u) - 1 <= 1 && (v46 = sub_34B9AF0((__int64)a2, *(_BYTE **)(a1 + 104), 0)) != 0 )
  {
    if ( (*((_WORD *)a2 + 1) & 3) != 2 )
    {
      v55 = (__int64 **)sub_B2D7E0(**(_QWORD **)(a1 + 48), "disable-tail-calls", 0x12u);
      v42 = sub_A72A30((__int64 *)&v55);
      v43 = v46;
      if ( v42 )
        v43 = 0;
      v46 = v43;
    }
  }
  else
  {
    v46 = 0;
  }
  v78 = v80;
  v56 = 0xFFFFFFFF00000020LL;
  v66 = v68;
  v67 = 0x1000000000LL;
  v69 = v71;
  v70 = 0x1000000000LL;
  v73 = 0x1000000000LL;
  v72 = v74;
  v76 = 0x400000000LL;
  v79 = 0x400000000LL;
  v27 = *((_QWORD *)a2 - 4);
  v57 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v75 = v77;
  v55 = v44;
  v58 = v27;
  v28 = sub_A74710((_QWORD *)a2 + 9, 0, 15);
  if ( !v28 )
  {
    v29 = *((_QWORD *)a2 - 4);
    if ( v29 )
    {
      if ( !*(_BYTE *)v29 && *(_QWORD *)(v29 + 24) == *((_QWORD *)a2 + 10) )
      {
        v48 = *(_QWORD *)(v29 + 120);
        v28 = sub_A74710(&v48, 0, 15);
      }
    }
  }
  LOBYTE(v56) = v56 & 0xF7 | (8 * (v28 & 1));
  v30 = sub_A73ED0((_QWORD *)a2 + 9, 36);
  if ( !v30 )
    v30 = sub_B49560((__int64)a2, 36);
  LOBYTE(v56) = v56 & 0xCB
              | ((32 * (*((_QWORD *)a2 + 2) != 0)) | (16 * v30) | (4 * (*(_DWORD *)(v47 + 8) >> 8 != 0))) & 0x34;
  v31 = sub_A74710((_QWORD *)a2 + 9, 0, 54);
  if ( !v31 )
  {
    v32 = *((_QWORD *)a2 - 4);
    if ( v32 )
    {
      if ( !*(_BYTE *)v32 && *(_QWORD *)(v32 + 24) == *((_QWORD *)a2 + 10) )
      {
        v48 = *(_QWORD *)(v32 + 120);
        v31 = sub_A74710(&v48, 0, 54);
      }
    }
  }
  LOBYTE(v56) = v56 & 0xFE | v31 & 1;
  v33 = sub_A74710((_QWORD *)a2 + 9, 0, 79);
  if ( !v33 )
  {
    v34 = *((_QWORD *)a2 - 4);
    if ( v34 )
    {
      if ( !*(_BYTE *)v34 && *(_QWORD *)(v34 + 24) == *((_QWORD *)a2 + 10) )
      {
        v48 = *(_QWORD *)(v34 + 120);
        v33 = sub_A74710(&v48, 0, 79);
      }
    }
  }
  v35 = (unsigned __int64)v60;
  LOBYTE(v56) = v56 & 0xFD | (2 * (v33 & 1));
  LODWORD(v57) = (*((_WORD *)a2 + 1) >> 2) & 0x3FF;
  v36 = v49;
  v49 = 0;
  v60 = v36;
  v37 = v50;
  v50 = 0;
  v61 = v37;
  v38 = v51;
  v51 = 0;
  v62 = v38;
  if ( v35 )
    j_j___libc_free_0(v35);
  v39 = *(_DWORD *)(v47 + 12);
  v63 = a2;
  HIDWORD(v56) = v39 - 1;
  BYTE1(v56) = v46;
  sub_B17DD0((__int64)a2);
  v40 = sub_3743390((_QWORD *)a1, &v55);
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
  if ( v75 != v77 )
    _libc_free((unsigned __int64)v75);
  if ( v72 != v74 )
    _libc_free((unsigned __int64)v72);
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  if ( v60 )
    j_j___libc_free_0((unsigned __int64)v60);
  if ( v49 )
    j_j___libc_free_0((unsigned __int64)v49);
  return v40;
}
