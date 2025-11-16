// Function: sub_101BE30
// Address: 0x101be30
//
__int64 __fastcall sub_101BE30(__int64 *a1, __int64 *a2, char a3, char a4, __m128i *a5, int a6)
{
  __int64 *v7; // r13
  unsigned __int8 v10; // al
  __int64 v11; // r15
  int v13; // eax
  __int64 *v14; // rdx
  __int64 *v15; // r14
  __int64 *v16; // rsi
  int v17; // eax
  unsigned __int8 **v18; // rdx
  unsigned __int8 *v19; // rdx
  int v20; // eax
  int v21; // eax
  unsigned __int8 **v22; // rcx
  unsigned __int8 *v23; // rax
  __int64 v24; // r14
  __int64 v25; // rsi
  unsigned __int8 *v26; // r14
  unsigned int v27; // eax
  __int64 *v28; // rax
  __int64 v29; // rcx
  int v30; // edx
  int v31; // eax
  __int64 v32; // rdi
  unsigned __int16 v33; // ax
  __int64 *v34; // r14
  __int64 *v35; // rax
  unsigned int v36; // r8d
  __int64 *v37; // r9
  __int64 *v38; // rax
  __int64 *v39; // r14
  __int64 *v40; // rax
  unsigned int v41; // r8d
  __int64 *v42; // r9
  __int64 *v43; // rax
  unsigned __int8 *v44; // rax
  __int64 *v45; // rsi
  __int64 *v46; // rdx
  unsigned __int8 *v47; // rsi
  __int64 *v48; // [rsp+8h] [rbp-A8h]
  __int64 *v49; // [rsp+8h] [rbp-A8h]
  unsigned int v50; // [rsp+10h] [rbp-A0h]
  unsigned int v51; // [rsp+10h] [rbp-A0h]
  unsigned int v52; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v55; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int8 *v56; // [rsp+28h] [rbp-88h] BYREF
  __int64 *v57; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v58; // [rsp+38h] [rbp-78h]
  __int64 v59; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v60; // [rsp+48h] [rbp-68h]
  __int64 *v61; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v62; // [rsp+58h] [rbp-58h]
  __int64 *v63; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v64; // [rsp+68h] [rbp-48h]
  __int64 v65[8]; // [rsp+70h] [rbp-40h] BYREF

  v7 = a2;
  v10 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x15u )
  {
    if ( *(_BYTE *)a2 <= 0x15u )
    {
      v11 = sub_96E6C0(0xFu, (__int64)a1, a2, a5->m128i_i64[0]);
      if ( v11 )
        return v11;
      v10 = *(_BYTE *)a1;
    }
    if ( v10 == 13 )
      return sub_ACADE0((__int64 **)a1[1]);
  }
  if ( *(_BYTE *)a2 == 13 )
    return sub_ACADE0((__int64 **)a1[1]);
  if ( (unsigned __int8)sub_1003090((__int64)a5, (unsigned __int8 *)a1)
    || (unsigned __int8)sub_1003090((__int64)a5, (unsigned __int8 *)a2) )
  {
    return sub_ACA8A0((__int64 **)a1[1]);
  }
  if ( (unsigned __int8)sub_FFFE90((__int64)a2) )
    return (__int64)a1;
  if ( a1 == a2 )
    return sub_AD6530(a1[1], (__int64)a2);
  if ( !(unsigned __int8)sub_FFFE90((__int64)a1) )
    goto LABEL_20;
  if ( a4 )
    return sub_AD6530(a1[1], (__int64)a2);
  sub_9AC330((__int64)&v63, (__int64)a2, 0, a5);
  if ( v64 > 0x40 )
  {
    a2 = v63;
    if ( (v63[(v64 - 1) >> 6] & (1LL << ((unsigned __int8)v64 - 1))) != 0 )
      goto LABEL_19;
    v50 = v64 - 1;
    if ( v50 != (unsigned int)sub_C445E0((__int64)&v63) )
      goto LABEL_19;
LABEL_69:
    if ( a3 )
      v7 = (__int64 *)sub_AD6530(a1[1], (__int64)a2);
    sub_969240(v65);
    v11 = (__int64)v7;
    sub_969240((__int64 *)&v63);
    return v11;
  }
  if ( v63 == (__int64 *)((1LL << ((unsigned __int8)v64 - 1)) - 1) )
    goto LABEL_69;
LABEL_19:
  sub_969240(v65);
  sub_969240((__int64 *)&v63);
LABEL_20:
  v13 = *(unsigned __int8 *)a1;
  if ( !a6 )
    goto LABEL_29;
  if ( (_BYTE)v13 == 42 )
  {
    v49 = (__int64 *)*(a1 - 8);
    if ( v49 )
    {
      v39 = (__int64 *)*(a1 - 4);
      if ( v39 )
      {
        v40 = (__int64 *)sub_101AFF0(15, (__int64 *)*(a1 - 4), v7, a5, a6 - 1);
        v41 = a6 - 1;
        v42 = v49;
        if ( v40 )
        {
          v11 = (__int64)sub_101AFF0(13, v49, v40, a5, v41);
          if ( v11 )
            return v11;
          v42 = v49;
          v41 = a6 - 1;
        }
        v52 = v41;
        v43 = (__int64 *)sub_101AFF0(15, v42, v7, a5, v41);
        if ( v43 )
        {
          v11 = (__int64)sub_101AFF0(13, v39, v43, a5, v52);
          if ( v11 )
            return v11;
        }
      }
    }
  }
  if ( *(_BYTE *)v7 != 42 )
  {
    if ( *(_BYTE *)v7 != 44 )
      goto LABEL_28;
    goto LABEL_24;
  }
  v34 = (__int64 *)*(v7 - 8);
  if ( v34 )
  {
    v48 = (__int64 *)*(v7 - 4);
    if ( v48 )
    {
      v35 = (__int64 *)sub_101AFF0(15, a1, v34, a5, a6 - 1);
      v36 = a6 - 1;
      v37 = v48;
      if ( v35 )
      {
        v11 = (__int64)sub_101AFF0(15, v35, v48, a5, v36);
        if ( v11 )
          return v11;
        v37 = v48;
        v36 = a6 - 1;
      }
      v51 = v36;
      v38 = (__int64 *)sub_101AFF0(15, a1, v37, a5, v36);
      if ( v38 )
      {
        v11 = (__int64)sub_101AFF0(15, v38, v34, a5, v51);
        if ( v11 )
          return v11;
      }
      if ( *(_BYTE *)v7 == 44 )
      {
LABEL_24:
        v14 = (__int64 *)*(v7 - 8);
        if ( v14 )
        {
          v15 = (__int64 *)*(v7 - 4);
          if ( v15 )
          {
            v16 = (__int64 *)sub_101AFF0(15, a1, v14, a5, a6 - 1);
            if ( v16 )
            {
              v11 = (__int64)sub_101AFF0(13, v16, v15, a5, a6 - 1);
              if ( v11 )
                return v11;
            }
          }
        }
      }
    }
  }
LABEL_28:
  v13 = *(unsigned __int8 *)a1;
  if ( (_BYTE)v13 == 67 )
  {
    v45 = (__int64 *)*(a1 - 4);
    if ( !v45 )
      goto LABEL_37;
    if ( *(_BYTE *)v7 != 67 )
      goto LABEL_37;
    v46 = (__int64 *)*(v7 - 4);
    if ( !v46 || v46[1] != v45[1] )
      goto LABEL_37;
    v47 = sub_101AFF0(15, v45, v46, a5, a6 - 1);
    if ( v47 )
    {
      v11 = sub_1001480(0x26u, v47, a1[1], a5->m128i_i64);
      if ( v11 )
        return v11;
    }
    v13 = *(unsigned __int8 *)a1;
  }
LABEL_29:
  if ( (unsigned __int8)v13 <= 0x1Cu )
  {
    if ( (_BYTE)v13 != 5 )
    {
LABEL_31:
      if ( !a6 )
        goto LABEL_32;
      v32 = a1[1];
      if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17 <= 1 )
        v32 = **(_QWORD **)(v32 + 16);
      if ( !sub_BCAC40(v32, 1) || (v11 = sub_101B6D0((unsigned __int8 *)a1, (unsigned __int8 *)v7, a5, a6 - 1)) == 0 )
      {
        if ( a6 != 3
          || (v33 = sub_9A1D50(0x20u, (__int64)a1, (unsigned __int8 *)v7, a5[2].m128i_i64[1], a5->m128i_i64[0]),
              !HIBYTE(v33))
          || !(_BYTE)v33
          || (v11 = sub_AD6530(a1[1], (__int64)a1)) == 0 )
        {
LABEL_32:
          v11 = 0;
          if ( a4 )
          {
            if ( *(_BYTE *)v7 != 59 )
              return 0;
            v11 = *(v7 - 8);
            if ( !v11 )
              return 0;
            v44 = (unsigned __int8 *)*(v7 - 4);
            if ( !v44 )
              return 0;
            if ( v44 != (unsigned __int8 *)a1 )
              return 0;
            v61 = 0;
            if ( !(unsigned __int8)sub_1004E80(&v61, (__int64)a1) )
              return 0;
          }
        }
      }
      return v11;
    }
    v17 = *((unsigned __int16 *)a1 + 1);
    goto LABEL_38;
  }
LABEL_37:
  v17 = v13 - 29;
LABEL_38:
  if ( v17 != 47 )
    goto LABEL_31;
  v18 = (*((_BYTE *)a1 + 7) & 0x40) != 0
      ? (unsigned __int8 **)*(a1 - 1)
      : (unsigned __int8 **)&a1[-4 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v19 = *v18;
  if ( !v19 )
    goto LABEL_31;
  v20 = *(unsigned __int8 *)v7;
  if ( (unsigned __int8)v20 > 0x1Cu )
  {
    v21 = v20 - 29;
  }
  else
  {
    if ( (_BYTE)v20 != 5 )
      goto LABEL_31;
    v21 = *((unsigned __int16 *)v7 + 1);
  }
  if ( v21 != 47 )
    goto LABEL_31;
  v22 = (*((_BYTE *)v7 + 7) & 0x40) != 0
      ? (unsigned __int8 **)*(v7 - 1)
      : (unsigned __int8 **)&v7[-4 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
  v23 = *v22;
  if ( !*v22 )
    goto LABEL_31;
  v24 = a5->m128i_i64[0];
  v55 = v19;
  v56 = v23;
  sub_FFF250((__int64)&v57, v24, &v55);
  v25 = v24;
  v26 = 0;
  sub_FFF250((__int64)&v59, v25, &v56);
  if ( v55 == v56 )
  {
    v62 = v58;
    if ( v58 > 0x40 )
      sub_C43780((__int64)&v61, (const void **)&v57);
    else
      v61 = v57;
    sub_C46B40((__int64)&v61, &v59);
    v27 = v62;
    v62 = 0;
    v64 = v27;
    v63 = v61;
    v28 = (__int64 *)sub_BD5C60((__int64)v55);
    v26 = (unsigned __int8 *)sub_ACCFD0(v28, (__int64)&v63);
    if ( v64 > 0x40 && v63 )
      j_j___libc_free_0_0(v63);
    if ( v62 > 0x40 && v61 )
      j_j___libc_free_0_0(v61);
    v29 = *((_QWORD *)v55 + 1);
    v30 = *(unsigned __int8 *)(v29 + 8);
    if ( (unsigned int)(v30 - 17) <= 1 )
    {
      v31 = *(_DWORD *)(v29 + 32);
      BYTE4(v63) = (_BYTE)v30 == 18;
      LODWORD(v63) = v31;
      v26 = (unsigned __int8 *)sub_AD5E10((__int64)v63, v26);
    }
  }
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( !v26 )
    goto LABEL_31;
  return sub_96F3F0((__int64)v26, a1[1], 1, a5->m128i_i64[0]);
}
