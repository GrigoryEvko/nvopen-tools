// Function: sub_D049D0
// Address: 0xd049d0
//
__int64 __fastcall sub_D049D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v9; // rax
  __int64 *v10; // r14
  __int64 v11; // r13
  unsigned __int64 v12; // rcx
  const void **v13; // rsi
  bool v14; // al
  __int32 v15; // edx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 *v19; // rbx
  unsigned __int32 v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  unsigned __int32 v25; // eax
  __int64 v26; // rdx
  unsigned int v27; // r15d
  __int64 v28; // r12
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rsi
  unsigned __int64 v32; // rdx
  bool v33; // zf
  unsigned __int64 v34; // rax
  unsigned __int32 v35; // edx
  __int64 v36; // r11
  bool v37; // al
  __int32 v38; // eax
  unsigned __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // [rsp-158h] [rbp-158h]
  unsigned __int64 v42; // [rsp-150h] [rbp-150h]
  unsigned __int32 v43; // [rsp-150h] [rbp-150h]
  bool v44; // [rsp-150h] [rbp-150h]
  __int64 v45; // [rsp-148h] [rbp-148h]
  const void **v47; // [rsp-140h] [rbp-140h]
  unsigned __int8 v48; // [rsp-140h] [rbp-140h]
  unsigned __int8 v49; // [rsp-140h] [rbp-140h]
  unsigned __int8 v50; // [rsp-140h] [rbp-140h]
  unsigned __int8 v51; // [rsp-140h] [rbp-140h]
  const void **v52; // [rsp-140h] [rbp-140h]
  unsigned __int8 v53; // [rsp-140h] [rbp-140h]
  unsigned __int8 v54; // [rsp-140h] [rbp-140h]
  unsigned __int8 v55; // [rsp-140h] [rbp-140h]
  unsigned __int8 v56; // [rsp-140h] [rbp-140h]
  unsigned __int8 v57; // [rsp-140h] [rbp-140h]
  bool v58; // [rsp-140h] [rbp-140h]
  __int64 v59; // [rsp-138h] [rbp-138h] BYREF
  unsigned __int32 v60; // [rsp-130h] [rbp-130h]
  __int64 v61; // [rsp-128h] [rbp-128h] BYREF
  unsigned __int32 v62; // [rsp-120h] [rbp-120h]
  __int64 v63; // [rsp-118h] [rbp-118h] BYREF
  unsigned int v64; // [rsp-110h] [rbp-110h]
  __int64 v65; // [rsp-108h] [rbp-108h] BYREF
  unsigned __int32 v66; // [rsp-100h] [rbp-100h]
  __int64 v67; // [rsp-F8h] [rbp-F8h] BYREF
  unsigned int v68; // [rsp-F0h] [rbp-F0h]
  __int64 v69; // [rsp-E8h] [rbp-E8h] BYREF
  unsigned __int32 v70; // [rsp-E0h] [rbp-E0h]
  __m128i v71; // [rsp-D8h] [rbp-D8h] BYREF
  int v72; // [rsp-C8h] [rbp-C8h]
  char v73; // [rsp-C4h] [rbp-C4h]
  __m128i v74; // [rsp-B8h] [rbp-B8h] BYREF
  int v75; // [rsp-A8h] [rbp-A8h]
  char v76; // [rsp-A4h] [rbp-A4h]
  const void *v77; // [rsp-A0h] [rbp-A0h] BYREF
  unsigned int v78; // [rsp-98h] [rbp-98h]
  __int64 v79; // [rsp-90h] [rbp-90h] BYREF
  unsigned __int32 v80; // [rsp-88h] [rbp-88h]
  __m128i v81; // [rsp-78h] [rbp-78h] BYREF
  int v82; // [rsp-68h] [rbp-68h]
  char v83; // [rsp-64h] [rbp-64h]
  const void *v84; // [rsp-60h] [rbp-60h] BYREF
  unsigned int v85; // [rsp-58h] [rbp-58h]
  __int64 v86; // [rsp-50h] [rbp-50h] BYREF
  unsigned int v87; // [rsp-48h] [rbp-48h]

  if ( *(_DWORD *)(a2 + 32) != 2 || a3 == -1 || a3 == 0xBFFFFFFFFFFFFFFELL || a4 == -1 )
    return 0;
  if ( a4 == 0xBFFFFFFFFFFFFFFELL )
    return 0;
  v81.m128i_i8[8] = (a3 & 0x4000000000000000LL) != 0;
  v81.m128i_i64[0] = a3 & 0x3FFFFFFFFFFFFFFFLL;
  v45 = sub_CA1930(&v81);
  v81.m128i_i64[0] = a4 & 0x3FFFFFFFFFFFFFFFLL;
  v81.m128i_i8[8] = (a4 & 0x4000000000000000LL) != 0;
  v9 = sub_CA1930(&v81);
  v10 = *(__int64 **)(a2 + 24);
  v11 = v9;
  if ( *((_DWORD *)v10 + 4) )
    return 0;
  v12 = *v10;
  if ( *(_QWORD *)(*v10 + 8) != *(_QWORD *)(v10[7] + 8) )
    return 0;
  if ( (v10[1] != v10[8] || *((_DWORD *)v10 + 18))
    && (!*((_BYTE *)v10 + 20) && !*((_BYTE *)v10 + 76)
     || *((_DWORD *)v10 + 2) + *((_DWORD *)v10 + 3) != *((_DWORD *)v10 + 16) + *((_DWORD *)v10 + 17)
     || *((_DWORD *)v10 + 18)) )
  {
    return 0;
  }
  v13 = (const void **)(v10 + 10);
  v47 = (const void **)(v10 + 3);
  if ( *((_BYTE *)v10 + 49) == *((_BYTE *)v10 + 105) )
  {
    v29 = *((_DWORD *)v10 + 22);
    v81.m128i_i32[2] = v29;
    if ( v29 > 0x40 )
    {
      sub_C43780((__int64)&v81, v13);
      v29 = v81.m128i_u32[2];
      if ( v81.m128i_i32[2] > 0x40u )
      {
        sub_C43D10((__int64)&v81);
        goto LABEL_94;
      }
      v30 = v81.m128i_i64[0];
    }
    else
    {
      v30 = v10[10];
    }
    v31 = ~v30;
    v32 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v29;
    v33 = v29 == 0;
    v34 = 0;
    if ( !v33 )
      v34 = v32;
    v81.m128i_i64[0] = v31 & v34;
LABEL_94:
    sub_C46250((__int64)&v81);
    v35 = v81.m128i_u32[2];
    v36 = v81.m128i_i64[0];
    v81.m128i_i32[2] = 0;
    v74.m128i_i32[2] = v35;
    v74.m128i_i64[0] = v81.m128i_i64[0];
    if ( *((_DWORD *)v10 + 8) <= 0x40u )
    {
      v37 = v10[3] == v81.m128i_i64[0];
    }
    else
    {
      v41 = v81.m128i_i64[0];
      v43 = v35;
      v37 = sub_C43C50((__int64)v47, (const void **)&v74);
      v36 = v41;
      v35 = v43;
    }
    if ( v35 > 0x40 )
    {
      if ( v36 )
      {
        v44 = v37;
        j_j___libc_free_0_0(v36);
        v37 = v44;
        if ( v81.m128i_i32[2] > 0x40u )
        {
          if ( v81.m128i_i64[0] )
          {
            j_j___libc_free_0_0(v81.m128i_i64[0]);
            v37 = v44;
          }
        }
      }
    }
    if ( !v37 )
      return 0;
    v12 = *v10;
    if ( *(_QWORD *)(v10[7] + 8) != *(_QWORD *)(*v10 + 8) )
      return 0;
    goto LABEL_19;
  }
  if ( *((_DWORD *)v10 + 8) > 0x40u )
  {
    v42 = *v10;
    v14 = sub_C43C50((__int64)v47, v13);
    v12 = v42;
    if ( v14 )
      goto LABEL_19;
    return 0;
  }
  if ( v10[3] != v10[10] )
    return 0;
LABEL_19:
  v81 = (__m128i)v12;
  v82 = 0;
  v83 = 0;
  sub_D02480(&v74, &v81, 0);
  v71 = (__m128i)(unsigned __int64)v10[7];
  v73 = 0;
  v72 = 0;
  sub_D02480(&v81, &v71, 0);
  if ( v78 <= 0x40 )
  {
    if ( v77 != v84 )
      goto LABEL_22;
  }
  else if ( !sub_C43C50((__int64)&v77, &v84) )
  {
    goto LABEL_22;
  }
  if ( *(_QWORD *)(v74.m128i_i64[0] + 8) == *(_QWORD *)(v81.m128i_i64[0] + 8)
    && (v74.m128i_i64[1] == v81.m128i_i64[1] && v75 == v82
     || (v76 || v83) && v74.m128i_i32[2] + v74.m128i_i32[3] == v81.m128i_i32[2] + v81.m128i_i32[3] && v75 == v82)
    && (unsigned __int8)sub_D04110(a1, v74.m128i_i64[0], v81.m128i_i64[0], a7) )
  {
    v71.m128i_i32[2] = v80;
    if ( v80 > 0x40 )
      sub_C43780((__int64)&v71, (const void **)&v79);
    else
      v71.m128i_i64[0] = v79;
    sub_C46B40((__int64)&v71, &v86);
    v15 = v71.m128i_i32[2];
    v16 = v71.m128i_i64[0];
    v60 = v71.m128i_u32[2];
    v59 = v71.m128i_i64[0];
    if ( v71.m128i_i32[2] > 0x40u )
    {
      sub_C43780((__int64)&v71, (const void **)&v59);
      v15 = v71.m128i_i32[2];
      if ( v71.m128i_i32[2] > 0x40u )
      {
        sub_C43D10((__int64)&v71);
        goto LABEL_47;
      }
      v16 = v71.m128i_i64[0];
    }
    v17 = ~v16;
    v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
    if ( !v15 )
      v18 = 0;
    v71.m128i_i64[0] = v18 & v17;
LABEL_47:
    v19 = &v61;
    sub_C46250((__int64)&v71);
    v62 = v71.m128i_u32[2];
    v61 = v71.m128i_i64[0];
    if ( (int)sub_C49970((__int64)&v59, (unsigned __int64 *)&v61) < 0 )
      v19 = &v59;
    if ( v60 <= 0x40 && *((_DWORD *)v19 + 2) <= 0x40u )
    {
      v40 = *v19;
      v60 = *((_DWORD *)v19 + 2);
      v59 = v40;
    }
    else
    {
      sub_C43990((__int64)&v59, (__int64)v19);
    }
    v20 = *((_DWORD *)v10 + 8);
    v21 = v10[3];
    v22 = 1LL << ((unsigned __int8)v20 - 1);
    if ( v20 > 0x40 )
    {
      if ( (*(_QWORD *)(v21 + 8LL * ((v20 - 1) >> 6)) & v22) == 0 )
      {
        v70 = *((_DWORD *)v10 + 8);
        sub_C43780((__int64)&v69, v47);
        goto LABEL_59;
      }
      v71.m128i_i32[2] = *((_DWORD *)v10 + 8);
      sub_C43780((__int64)&v71, v47);
      v20 = v71.m128i_u32[2];
      if ( v71.m128i_i32[2] > 0x40u )
      {
        sub_C43D10((__int64)&v71);
LABEL_58:
        sub_C46250((__int64)&v71);
        v70 = v71.m128i_u32[2];
        v69 = v71.m128i_i64[0];
LABEL_59:
        sub_C44AB0((__int64)&v71, (__int64)&v59, *((_DWORD *)v10 + 8));
        sub_C472A0((__int64)&v63, (__int64)&v71, &v69);
        if ( v71.m128i_i32[2] > 0x40u && v71.m128i_i64[0] )
          j_j___libc_free_0_0(v71.m128i_i64[0]);
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        v24 = *(_QWORD *)(a2 + 8);
        v52 = (const void **)(a2 + 8);
        v25 = *(_DWORD *)(a2 + 16);
        v26 = 1LL << ((unsigned __int8)v25 - 1);
        if ( v25 > 0x40 )
        {
          if ( (*(_QWORD *)(v24 + 8LL * ((v25 - 1) >> 6)) & v26) == 0 )
          {
            v66 = *(_DWORD *)(a2 + 16);
            sub_C43780((__int64)&v65, v52);
            goto LABEL_68;
          }
          v71.m128i_i32[2] = *(_DWORD *)(a2 + 16);
          sub_C43780((__int64)&v71, v52);
          v25 = v71.m128i_u32[2];
          if ( v71.m128i_i32[2] > 0x40u )
          {
            sub_C43D10((__int64)&v71);
LABEL_122:
            sub_C46250((__int64)&v71);
            v66 = v71.m128i_u32[2];
            v65 = v71.m128i_i64[0];
            goto LABEL_68;
          }
        }
        else
        {
          if ( (v26 & v24) == 0 )
          {
            v66 = *(_DWORD *)(a2 + 16);
            v65 = v24;
LABEL_68:
            sub_C46A40((__int64)&v65, v45);
            v27 = v66;
            v28 = v65;
            v66 = 0;
            v68 = v27;
            v67 = v65;
            if ( (int)sub_C49970((__int64)&v63, (unsigned __int64 *)&v67) >= 0 )
            {
              sub_9692E0((__int64)&v69, (__int64 *)v52);
              sub_C46A40((__int64)&v69, v11);
              v38 = v70;
              v70 = 0;
              v71.m128i_i32[2] = v38;
              v71.m128i_i64[0] = v69;
              v58 = (int)sub_C49970((__int64)&v63, (unsigned __int64 *)&v71) >= 0;
              sub_969240(v71.m128i_i64);
              sub_969240(&v69);
              result = v58;
            }
            else
            {
              result = 0;
            }
            if ( v27 > 0x40 && v28 )
            {
              v53 = result;
              j_j___libc_free_0_0(v28);
              result = v53;
            }
            if ( v66 > 0x40 && v65 )
            {
              v54 = result;
              j_j___libc_free_0_0(v65);
              result = v54;
            }
            if ( v64 > 0x40 && v63 )
            {
              v55 = result;
              j_j___libc_free_0_0(v63);
              result = v55;
            }
            if ( v62 > 0x40 && v61 )
            {
              v56 = result;
              j_j___libc_free_0_0(v61);
              result = v56;
            }
            if ( v60 > 0x40 && v59 )
            {
              v57 = result;
              j_j___libc_free_0_0(v59);
              result = v57;
            }
            goto LABEL_23;
          }
          v71.m128i_i32[2] = *(_DWORD *)(a2 + 16);
          v71.m128i_i64[0] = v24;
        }
        v39 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v25;
        if ( !v25 )
          v39 = 0;
        v71.m128i_i64[0] = v39 & ~v71.m128i_i64[0];
        goto LABEL_122;
      }
    }
    else
    {
      if ( (v22 & v21) == 0 )
      {
        v70 = *((_DWORD *)v10 + 8);
        v69 = v21;
        goto LABEL_59;
      }
      v71.m128i_i32[2] = *((_DWORD *)v10 + 8);
      v71.m128i_i64[0] = v21;
    }
    v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v20;
    if ( !v20 )
      v23 = 0;
    v71.m128i_i64[0] = v23 & ~v71.m128i_i64[0];
    goto LABEL_58;
  }
LABEL_22:
  result = 0;
LABEL_23:
  if ( v87 > 0x40 && v86 )
  {
    v48 = result;
    j_j___libc_free_0_0(v86);
    result = v48;
  }
  if ( v85 > 0x40 && v84 )
  {
    v49 = result;
    j_j___libc_free_0_0(v84);
    result = v49;
  }
  if ( v80 > 0x40 && v79 )
  {
    v50 = result;
    j_j___libc_free_0_0(v79);
    result = v50;
  }
  if ( v78 > 0x40 )
  {
    if ( v77 )
    {
      v51 = result;
      j_j___libc_free_0_0(v77);
      return v51;
    }
  }
  return result;
}
