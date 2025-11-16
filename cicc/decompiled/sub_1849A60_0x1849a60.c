// Function: sub_1849A60
// Address: 0x1849a60
//
__int64 __fastcall sub_1849A60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 i; // r12
  unsigned __int8 v7; // al
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // r10
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  int v21; // esi
  unsigned int v22; // edx
  __int64 v23; // r11
  int v24; // ecx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // r15
  unsigned __int8 v30; // al
  __int64 v31; // rsi
  char v32; // al
  char v33; // r15
  char v34; // cl
  __int64 v35; // r10
  unsigned __int8 v36; // al
  int v37; // r15d
  int v38; // esi
  __int64 **v39; // r11
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r10
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // r10
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 **v56; // r13
  __int64 v57; // rbx
  __int64 **v58; // r12
  __int64 *v59; // r14
  __int64 v60; // rdx
  char v61; // al
  __m128i v62; // xmm0
  unsigned __int8 v63; // cl
  __int64 v64; // [rsp+0h] [rbp-D0h]
  __int64 v65; // [rsp+0h] [rbp-D0h]
  __int64 v66; // [rsp+0h] [rbp-D0h]
  __int64 v67; // [rsp+8h] [rbp-C8h]
  __int64 v68; // [rsp+8h] [rbp-C8h]
  __int64 v69; // [rsp+8h] [rbp-C8h]
  __int64 v70; // [rsp+18h] [rbp-B8h]
  char v71; // [rsp+20h] [rbp-B0h]
  __int64 v72; // [rsp+20h] [rbp-B0h]
  __int64 v73; // [rsp+20h] [rbp-B0h]
  __int64 v74; // [rsp+20h] [rbp-B0h]
  __int64 v75; // [rsp+30h] [rbp-A0h]
  __int64 v76; // [rsp+30h] [rbp-A0h]
  __int64 v77; // [rsp+30h] [rbp-A0h]
  int v78; // [rsp+30h] [rbp-A0h]
  int v79; // [rsp+30h] [rbp-A0h]
  char v80; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v83; // [rsp+4Fh] [rbp-81h]
  __m128i v84; // [rsp+50h] [rbp-80h] BYREF
  __int64 v85; // [rsp+60h] [rbp-70h]
  __m128i v86[2]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v87; // [rsp+90h] [rbp-40h]

  v3 = a1 + 72;
  v4 = *(_QWORD *)(a1 + 80);
  if ( a1 + 72 == v4 )
    return 0;
  if ( !v4 )
    BUG();
  while ( 1 )
  {
    i = *(_QWORD *)(v4 + 24);
    if ( i != v4 + 16 )
      break;
    v4 = *(_QWORD *)(v4 + 8);
    if ( v3 == v4 )
      return 0;
    if ( !v4 )
      BUG();
  }
  if ( v4 == v3 )
    return 0;
  v83 = 0;
  v70 = a3 + 16;
  while ( 1 )
  {
    if ( !i )
      BUG();
    v7 = *(_BYTE *)(i - 8);
    v8 = i - 24;
    if ( v7 <= 0x17u )
    {
LABEL_41:
      if ( v7 == 55 )
      {
        if ( (*(_BYTE *)(i - 6) & 1) != 0 )
          goto LABEL_16;
        sub_141EDF0(v86, i - 24);
      }
      else
      {
        if ( v7 != 82 )
        {
LABEL_16:
          if ( (unsigned __int8)sub_15F3040(i - 24) )
            return 2;
          v83 |= sub_15F2ED0(i - 24);
          goto LABEL_18;
        }
        sub_141F0A0(v86, i - 24);
      }
      goto LABEL_44;
    }
    if ( v7 == 78 )
    {
      v9 = v8 | 4;
      goto LABEL_26;
    }
    if ( v7 == 29 )
      break;
    if ( v7 != 54 )
      goto LABEL_41;
    if ( (*(_BYTE *)(i - 6) & 1) != 0 )
      goto LABEL_16;
    sub_141EB40(v86, (__int64 *)(i - 24));
LABEL_44:
    if ( !(unsigned __int8)sub_134CBB0(a2, (__int64)v86, 1u) )
      goto LABEL_16;
LABEL_18:
    for ( i = *(_QWORD *)(i + 8); i == v4 - 24 + 40; i = *(_QWORD *)(v4 + 24) )
    {
      v4 = *(_QWORD *)(v4 + 8);
      if ( v3 == v4 )
        return v83;
      if ( !v4 )
        BUG();
    }
    if ( v3 == v4 )
      return v83;
  }
  v9 = v8 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_26:
  v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_16;
  v11 = v9 >> 2;
  v12 = *(_BYTE *)(v10 + 23);
  v71 = v11 & 1;
  if ( (v11 & 1) != 0 )
  {
    if ( v12 >= 0
      || ((v75 = v10, v13 = sub_1648A40(v10), v10 = v75, v15 = v13 + v14, *(char *)(v75 + 23) >= 0)
        ? (v17 = v15 >> 4)
        : (v16 = sub_1648A40(v75), v10 = v75, v17 = (v15 - v16) >> 4),
          !(_DWORD)v17) )
    {
      v18 = (__int64 *)(v10 - 24);
      goto LABEL_33;
    }
  }
  else if ( v12 >= 0
         || ((v76 = v10, v25 = sub_1648A40(v10), v10 = v76, v27 = v25 + v26, *(char *)(v76 + 23) >= 0)
           ? (v29 = v27 >> 4)
           : (v28 = sub_1648A40(v76), v10 = v76, v29 = (v27 - v28) >> 4),
             !(_DWORD)v29) )
  {
    v18 = (__int64 *)(v10 - 72);
LABEL_33:
    v19 = *v18;
    if ( !*(_BYTE *)(v19 + 16) )
    {
      if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
      {
        v20 = v70;
        v21 = 7;
      }
      else
      {
        v20 = *(_QWORD *)(a3 + 16);
        v38 = *(_DWORD *)(a3 + 24);
        if ( !v38 )
          goto LABEL_50;
        v21 = v38 - 1;
      }
      v22 = v21 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v23 = *(_QWORD *)(v20 + 8LL * v22);
      if ( v23 == v19 )
        goto LABEL_18;
      v24 = 1;
      while ( v23 != -8 )
      {
        v22 = v21 & (v24 + v22);
        v23 = *(_QWORD *)(v20 + 8LL * v22);
        if ( v23 == v19 )
          goto LABEL_18;
        ++v24;
      }
    }
  }
LABEL_50:
  v30 = *(_BYTE *)(v10 + 16);
  v31 = 0;
  if ( v30 > 0x17u )
  {
    if ( v30 == 78 )
    {
      v31 = v10 | 4;
    }
    else if ( v30 == 29 )
    {
      v31 = v10;
    }
  }
  v77 = v10;
  v32 = sub_134CC90(a2, v31);
  v33 = v32;
  v34 = v32;
  if ( (v32 & 3) == 0 )
    goto LABEL_18;
  v35 = v77;
  if ( (v32 & 0x30) == 0 )
  {
    v39 = (__int64 **)(v77 - 24LL * (*(_DWORD *)(v77 + 20) & 0xFFFFFFF));
    v40 = *(_BYTE *)(v77 + 23);
    if ( v71 )
    {
      if ( v40 < 0 )
      {
        v67 = v77 - 24LL * (*(_DWORD *)(v77 + 20) & 0xFFFFFFF);
        v41 = sub_1648A40(v77);
        v35 = v77;
        v34 = v33;
        v39 = (__int64 **)v67;
        if ( *(char *)(v77 + 23) >= 0 )
        {
          if ( (unsigned int)((v41 + v42) >> 4) )
LABEL_108:
            BUG();
        }
        else
        {
          v64 = v41 + v42;
          v43 = sub_1648A40(v77);
          v35 = v77;
          v34 = v33;
          v39 = (__int64 **)v67;
          if ( (unsigned int)((v64 - v43) >> 4) )
          {
            if ( *(char *)(v77 + 23) >= 0 )
              goto LABEL_108;
            v72 = v77;
            v44 = sub_1648A40(v77);
            v45 = v77;
            v78 = *(_DWORD *)(v44 + 8);
            if ( *(char *)(v45 + 23) >= 0 )
              BUG();
            v46 = sub_1648A40(v72);
            v39 = (__int64 **)v67;
            v34 = v33;
            v35 = v72;
            v48 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v46 + v47 - 4) - v78);
            goto LABEL_82;
          }
        }
      }
      v48 = -24;
    }
    else
    {
      if ( v40 < 0 )
      {
        v68 = v77 - 24LL * (*(_DWORD *)(v77 + 20) & 0xFFFFFFF);
        v49 = sub_1648A40(v77);
        v35 = v77;
        v34 = v33;
        v39 = (__int64 **)v68;
        if ( *(char *)(v77 + 23) >= 0 )
        {
          if ( (unsigned int)((v49 + v50) >> 4) )
LABEL_104:
            BUG();
        }
        else
        {
          v65 = v49 + v50;
          v51 = sub_1648A40(v77);
          v35 = v77;
          v34 = v33;
          v39 = (__int64 **)v68;
          if ( (unsigned int)((v65 - v51) >> 4) )
          {
            if ( *(char *)(v77 + 23) >= 0 )
              goto LABEL_104;
            v73 = v77;
            v52 = sub_1648A40(v77);
            v53 = v77;
            v79 = *(_DWORD *)(v52 + 8);
            if ( *(char *)(v53 + 23) >= 0 )
              BUG();
            v54 = sub_1648A40(v73);
            v39 = (__int64 **)v68;
            v34 = v33;
            v35 = v73;
            v48 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v54 + v55 - 4) - v79);
            goto LABEL_82;
          }
        }
      }
      v48 = -72;
    }
LABEL_82:
    if ( (__int64 **)(v35 + v48) != v39 )
    {
      v74 = v3;
      v56 = (__int64 **)(v35 + v48);
      v69 = v4;
      v57 = i - 24;
      v66 = i;
      v58 = v39;
      v80 = v34;
      do
      {
        v59 = *v58;
        v60 = **v58;
        v61 = *(_BYTE *)(v60 + 8);
        if ( v61 == 16 )
          v61 = *(_BYTE *)(**(_QWORD **)(v60 + 16) + 8LL);
        if ( v61 == 15 )
        {
          v84 = 0u;
          v85 = 0;
          sub_14A8180(v57, v84.m128i_i64, 0);
          v62 = _mm_loadu_si128(&v84);
          v86[0].m128i_i64[0] = (__int64)v59;
          v86[0].m128i_i64[1] = -1;
          v87 = v85;
          v86[1] = v62;
          if ( !(unsigned __int8)sub_134CBB0(a2, (__int64)v86, 1u) )
          {
            if ( (v80 & 2) != 0 )
              return 2;
            v63 = v83;
            if ( (v33 & 1) != 0 )
              v63 = v33 & 1;
            v83 = v63;
          }
        }
        v58 += 3;
      }
      while ( v58 != v56 );
      v3 = v74;
      v4 = v69;
      i = v66;
    }
    goto LABEL_18;
  }
  if ( (v32 & 2) == 0 )
  {
    v36 = v83;
    v37 = v33 & 1;
    if ( v37 )
      v36 = v37;
    v83 = v36;
    goto LABEL_18;
  }
  return 2;
}
