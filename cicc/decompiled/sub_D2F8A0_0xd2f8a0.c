// Function: sub_D2F8A0
// Address: 0xd2f8a0
//
char __fastcall sub_D2F8A0(
        unsigned __int8 *a1,
        unsigned __int8 a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9,
        int a10)
{
  int v10; // r15d
  char result; // al
  unsigned __int64 *v13; // r13
  __int64 v14; // r12
  unsigned __int8 **v16; // rax
  char v17; // dl
  unsigned __int8 v18; // al
  __int16 v19; // ax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdx
  _QWORD *v22; // rax
  int v23; // eax
  __int64 v24; // rdi
  char v25; // al
  unsigned int v26; // edx
  __int64 v27; // rsi
  _QWORD *v28; // rax
  bool v29; // al
  __int32 v30; // eax
  unsigned __int8 v31; // al
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rsi
  __int64 v37; // [rsp+0h] [rbp-120h]
  unsigned __int64 v38; // [rsp+8h] [rbp-118h]
  __int32 v39; // [rsp+8h] [rbp-118h]
  bool v40; // [rsp+8h] [rbp-118h]
  bool v41; // [rsp+8h] [rbp-118h]
  int v42; // [rsp+18h] [rbp-108h]
  __int64 v43; // [rsp+18h] [rbp-108h]
  __int64 v44; // [rsp+18h] [rbp-108h]
  unsigned int v45; // [rsp+18h] [rbp-108h]
  unsigned __int8 *v46; // [rsp+20h] [rbp-100h] BYREF
  int v47; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v48; // [rsp+2Fh] [rbp-F1h] BYREF
  bool v49; // [rsp+37h] [rbp-E9h] BYREF
  __int64 v50; // [rsp+38h] [rbp-E8h] BYREF
  int v51; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v52; // [rsp+48h] [rbp-D8h]
  __int64 v53; // [rsp+50h] [rbp-D0h]
  __int64 v54; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v55; // [rsp+68h] [rbp-B8h]
  __int64 v56; // [rsp+70h] [rbp-B0h]
  __int64 v57; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v58; // [rsp+88h] [rbp-98h]
  __m128i v59; // [rsp+A0h] [rbp-80h] BYREF
  int *v60; // [rsp+B0h] [rbp-70h]
  __int64 *v61; // [rsp+B8h] [rbp-68h]
  bool *v62; // [rsp+C0h] [rbp-60h]
  unsigned __int8 *v63; // [rsp+C8h] [rbp-58h]
  unsigned __int64 *v64; // [rsp+D0h] [rbp-50h]
  __int64 v65; // [rsp+D8h] [rbp-48h]
  __int16 v66; // [rsp+E0h] [rbp-40h]

  v10 = a10;
  v48 = a2;
  v46 = (unsigned __int8 *)a5;
  if ( !a10 )
    return 0;
  v13 = a3;
  v14 = a4;
  if ( !*(_BYTE *)(a9 + 28) )
    goto LABEL_10;
  v16 = *(unsigned __int8 ***)(a9 + 8);
  a4 = *(unsigned int *)(a9 + 20);
  a3 = (unsigned __int64 *)&v16[a4];
  if ( v16 != (unsigned __int8 **)a3 )
  {
    while ( *v16 != a1 )
    {
      if ( a3 == (unsigned __int64 *)++v16 )
        goto LABEL_41;
    }
    return 0;
  }
LABEL_41:
  v28 = (_QWORD *)a9;
  if ( (unsigned int)a4 < *(_DWORD *)(a9 + 16) )
  {
    *(_DWORD *)(a9 + 20) = a4 + 1;
    *a3 = (unsigned __int64)a1;
    ++*v28;
  }
  else
  {
LABEL_10:
    sub_C8CC70(a9, (__int64)a1, (__int64)a3, a4, a5, a6);
    if ( !v17 )
      return 0;
  }
  v47 = v10 - 1;
  v18 = *a1;
  if ( *a1 > 0x1Cu )
  {
    if ( v18 == 63 )
      goto LABEL_32;
    if ( v18 == 78 )
    {
LABEL_62:
      v24 = *((_QWORD *)a1 - 4);
      if ( *(_BYTE *)(*(_QWORD *)(v24 + 8) + 8LL) == 14 )
        return sub_D2F8A0(v24, v48, (_DWORD)v13, v14, (_DWORD)v46, a6, (__int64)a7, a8, a9, v47);
LABEL_15:
      v20 = sub_BD4FF0(a1, v14, &v54, &v57);
      v21 = v20;
      if ( *((_DWORD *)v13 + 2) <= 0x40u )
      {
        v22 = (_QWORD *)*v13;
      }
      else
      {
        v38 = v20;
        v42 = *((_DWORD *)v13 + 2);
        if ( v42 - (unsigned int)sub_C444A0((__int64)v13) > 0x40 )
          goto LABEL_65;
        v21 = v38;
        v22 = *(_QWORD **)*v13;
      }
      if ( v21 >= (unsigned __int64)v22 && !(_BYTE)v57 )
      {
        if ( !(_BYTE)v54 )
          goto LABEL_22;
        v59 = (__m128i)(unsigned __int64)v14;
        v61 = a7;
        v60 = 0;
        v62 = (bool *)a6;
        v63 = v46;
        v64 = 0;
        v65 = 0;
        v66 = 257;
        if ( (unsigned __int8)sub_9B6260((__int64)a1, &v59, 0) )
        {
LABEL_22:
          v23 = *a1;
          if ( (unsigned __int8)v23 <= 0x1Cu || (_BYTE)v23 == 60 )
            return v48 <= (unsigned __int8)sub_BD5420(a1, v14);
          if ( !v46 )
            goto LABEL_71;
          if ( (unsigned __int8)sub_98CF40((__int64)a1, (__int64)v46, (__int64)a7, 0) )
            return v48 <= (unsigned __int8)sub_BD5420(a1, v14);
        }
      }
LABEL_65:
      v23 = *a1;
      if ( (unsigned __int8)v23 <= 0x1Cu )
        goto LABEL_66;
LABEL_71:
      if ( (unsigned __int8)(v23 - 34) > 0x33u || (v32 = 0x8000000000041LL, !_bittest64(&v32, (unsigned int)(v23 - 34))) )
      {
LABEL_72:
        if ( (_BYTE)v23 != 79 )
          goto LABEL_68;
        goto LABEL_30;
      }
      v33 = sub_98AC40((__int64)a1, 1);
      LODWORD(v24) = v33;
      if ( v33 )
        return sub_D2F8A0(v24, v48, (_DWORD)v13, v14, (_DWORD)v46, a6, (__int64)a7, a8, a9, v47);
      if ( !(unsigned __int8)sub_D62CA0(a1, &v54, v14, a8, 0x10000, 0) )
      {
LABEL_78:
        LOBYTE(v23) = *a1;
        if ( *a1 == 85 )
        {
          v34 = *((_QWORD *)a1 - 4);
          if ( v34
            && !*(_BYTE *)v34
            && *(_QWORD *)(v34 + 24) == *((_QWORD *)a1 + 10)
            && (*(_BYTE *)(v34 + 33) & 0x20) != 0
            && *(_DWORD *)(v34 + 36) == 149 )
          {
            v44 = (__int64)a7;
            v35 = sub_B5B890((__int64)a1);
            return sub_D2F8A0(v35, v48, (_DWORD)v13, v14, (_DWORD)v46, a6, v44, a8, a9, v47);
          }
          goto LABEL_68;
        }
        if ( (unsigned __int8)v23 <= 0x1Cu )
        {
LABEL_66:
          if ( (_BYTE)v23 != 5 || *((_WORD *)a1 + 1) != 50 )
          {
LABEL_68:
            if ( v46 && !(unsigned __int8)sub_BD4ED0((__int64)a1) )
            {
              v51 = 0;
              v52 = 0;
              v53 = 0;
              LODWORD(v54) = 0;
              v55 = 0;
              v56 = 0;
              v31 = sub_BD5420(a1, v14);
              v61 = &v54;
              v59.m128i_i64[0] = (__int64)&v46;
              v59.m128i_i64[1] = (__int64)&a7;
              v60 = &v51;
              v62 = &v49;
              v63 = &v48;
              v50 = 0x560000005ALL;
              v49 = v31 >= v48;
              v64 = v13;
              sub_CF9460(
                (__int64)&v57,
                (__int64)a1,
                &v50,
                2,
                a6,
                (__int64)&v50,
                (unsigned __int8 (__fastcall *)(__int64, __int64, unsigned __int64, __int64, __int64, __int64, __int64, __int64, __int64))sub_D2F780,
                (__int64)&v59);
              return (_DWORD)v57 != 0;
            }
            return 0;
          }
          goto LABEL_30;
        }
        goto LABEL_72;
      }
      v36 = v54;
      v58 = *((_DWORD *)v13 + 2);
      if ( v58 > 0x40 )
      {
        sub_C43690((__int64)&v57, v54, 0);
        if ( v58 > 0x40 )
        {
          v45 = v58;
          if ( v45 != (unsigned int)sub_C444A0((__int64)&v57) )
            goto LABEL_90;
          goto LABEL_91;
        }
        v36 = v57;
      }
      else
      {
        v57 = v54;
      }
      if ( v36 )
      {
LABEL_90:
        if ( (int)sub_C49970((__int64)&v57, v13) >= 0 )
        {
          v59 = (__m128i)(unsigned __int64)v14;
          v61 = a7;
          v60 = 0;
          v62 = (bool *)a6;
          v63 = v46;
          v64 = 0;
          v65 = 0;
          v66 = 257;
          if ( (unsigned __int8)sub_9B6260((__int64)a1, &v59, 0) )
          {
            if ( !(unsigned __int8)sub_BD4ED0((__int64)a1) )
            {
              LOBYTE(v47) = (unsigned __int8)sub_BD5420(a1, v14) >= v48;
              sub_969240(&v57);
              return v47;
            }
          }
        }
      }
LABEL_91:
      if ( v58 > 0x40 && v57 )
        j_j___libc_free_0_0(v57);
      goto LABEL_78;
    }
    if ( v18 != 86 )
      goto LABEL_15;
    if ( (unsigned __int8)sub_D2F8A0(
                            *((_QWORD *)a1 - 8),
                            v48,
                            (_DWORD)v13,
                            v14,
                            (_DWORD)v46,
                            a6,
                            (__int64)a7,
                            a8,
                            a9,
                            v47) )
    {
LABEL_30:
      v24 = *((_QWORD *)a1 - 4);
      return sub_D2F8A0(v24, v48, (_DWORD)v13, v14, (_DWORD)v46, a6, (__int64)a7, a8, a9, v47);
    }
    return 0;
  }
  if ( v18 != 5 )
    goto LABEL_15;
  v19 = *((_WORD *)a1 + 1);
  if ( v19 != 34 )
  {
    if ( v19 != 49 )
      goto LABEL_15;
    goto LABEL_62;
  }
LABEL_32:
  v43 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  LODWORD(v55) = sub_AE43F0(v14, *((_QWORD *)a1 + 1));
  if ( (unsigned int)v55 > 0x40 )
    sub_C43690((__int64)&v54, 0, 0);
  else
    v54 = 0;
  v25 = sub_BB6360((__int64)a1, v14, (__int64)&v54, 0, 0);
  v26 = v55;
  if ( !v25 )
    goto LABEL_37;
  v27 = 1LL << ((unsigned __int8)v55 - 1);
  if ( (unsigned int)v55 > 0x40 )
  {
    if ( (*(_QWORD *)(v54 + 8LL * ((unsigned int)(v55 - 1) >> 6)) & v27) != 0 )
      goto LABEL_37;
    v58 = v55;
    sub_C43690((__int64)&v57, 1LL << v48, 0);
  }
  else
  {
    if ( (v27 & v54) != 0 )
      goto LABEL_37;
    v58 = v55;
    v57 = 1LL << v48;
  }
  sub_C4B490((__int64)&v59, (__int64)&v54, (__int64)&v57);
  if ( v59.m128i_i32[2] <= 0x40u )
  {
    v29 = v59.m128i_i64[0] != 0;
  }
  else
  {
    v39 = v59.m128i_i32[2];
    v29 = v39 != (unsigned int)sub_C444A0((__int64)&v59);
    if ( v59.m128i_i64[0] )
    {
      v40 = v29;
      j_j___libc_free_0_0(v59.m128i_i64[0]);
      v29 = v40;
    }
  }
  if ( v58 > 0x40 && v57 )
  {
    v41 = v29;
    j_j___libc_free_0_0(v57);
    v29 = v41;
  }
  v26 = v55;
  if ( !v29 )
  {
    v37 = (__int64)a7;
    sub_C44B10((__int64)&v57, (char **)v13, v55);
    sub_C45EE0((__int64)&v57, &v54);
    v30 = v58;
    v58 = 0;
    v59.m128i_i32[2] = v30;
    v59.m128i_i64[0] = v57;
    result = sub_D2F8A0(v43, v48, (unsigned int)&v59, v14, (_DWORD)v46, a6, v37, a8, a9, v47);
    if ( v59.m128i_i32[2] > 0x40u && v59.m128i_i64[0] )
    {
      LOBYTE(v47) = result;
      j_j___libc_free_0_0(v59.m128i_i64[0]);
      result = v47;
    }
    if ( v58 > 0x40 && v57 )
    {
      LOBYTE(v47) = result;
      j_j___libc_free_0_0(v57);
      result = v47;
    }
    v26 = v55;
    goto LABEL_38;
  }
LABEL_37:
  result = 0;
LABEL_38:
  if ( v26 > 0x40 && v54 )
  {
    LOBYTE(v47) = result;
    j_j___libc_free_0_0(v54);
    return v47;
  }
  return result;
}
