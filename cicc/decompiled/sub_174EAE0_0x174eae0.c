// Function: sub_174EAE0
// Address: 0x174eae0
//
__int64 __fastcall sub_174EAE0(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rdi
  __int64 result; // rax
  __int64 **v13; // r15
  __int64 v14; // r14
  __int64 v15; // rbx
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int64 v18; // rax
  unsigned __int8 v19; // dl
  __int64 v20; // rbx
  char v21; // al
  __int64 v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // rdi
  __int64 **v25; // rcx
  unsigned __int8 *v26; // rax
  __int64 v27; // rdi
  __int64 **v28; // rcx
  __int64 *v29; // r14
  unsigned __int8 *v30; // rax
  __int64 v31; // r14
  int v32; // ebx
  __int64 v33; // rbx
  __int64 v34; // r13
  _QWORD *v35; // rax
  double v36; // xmm4_8
  double v37; // xmm5_8
  __int64 ****v38; // rax
  int v39; // ebx
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // r12
  __int64 v44; // rdx
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 *v47; // rax
  int v48; // r14d
  int v49; // eax
  _QWORD *v50; // rdx
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 v53; // r14
  const char *v54; // rax
  __int64 v55; // rdx
  __int64 *v56; // rax
  int v57; // [rsp+8h] [rbp-98h]
  __int64 ***v58; // [rsp+8h] [rbp-98h]
  __int64 **v59; // [rsp+8h] [rbp-98h]
  __int64 *v60; // [rsp+18h] [rbp-88h] BYREF
  __int64 v61; // [rsp+20h] [rbp-80h] BYREF
  __int64 v62; // [rsp+28h] [rbp-78h] BYREF
  __int64 v63[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v64; // [rsp+40h] [rbp-60h]
  _QWORD *v65[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v66; // [rsp+60h] [rbp-40h]

  v11 = *(_QWORD *)(a2 + 8);
  if ( v11 && !*(_QWORD *)(v11 + 8) && *((_BYTE *)sub_1648700(v11) + 16) == 60 )
    return 0;
  result = sub_174B490(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( !result )
  {
    v13 = *(__int64 ***)a2;
    v14 = *(_QWORD *)(a2 - 24);
    v15 = *(_QWORD *)v14;
    if ( (*(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16
       || (unsigned __int8)sub_1705440((__int64)a1, *(_QWORD *)v14, *(_QWORD *)a2))
      && sub_1749C90(v14, (__int64)v13) )
    {
      v31 = (__int64)sub_174BF40(a1, v14, v13, 1u);
      v32 = sub_16431D0(v15);
      v57 = sub_16431D0((__int64)v13);
      if ( v57 - v32 < (unsigned int)sub_14C23D0(v31, a1[333], 0, a1[330], a2, a1[332]) )
      {
        v33 = *(_QWORD *)(a2 + 8);
        if ( v33 )
        {
          v34 = *a1;
          do
          {
            v35 = sub_1648700(v33);
            sub_170B990(v34, (__int64)v35);
            v33 = *(_QWORD *)(v33 + 8);
          }
          while ( v33 );
          if ( a2 == v31 )
            v31 = sub_1599EF0(*(__int64 ***)a2);
          sub_164D160(a2, v31, a3, a4, a5, a6, v36, v37, a9, a10);
          return a2;
        }
        return 0;
      }
      v46 = sub_15A0680((__int64)v13, (unsigned int)(v57 - v32), 0);
      v42 = a1[1];
      v43 = v46;
      v64 = 259;
      LOWORD(v66) = 257;
      v44 = v46;
      v45 = v31;
      v63[0] = (__int64)"sext";
LABEL_39:
      v47 = (__int64 *)sub_173DC60(v42, v45, v44, v63, 0, 0, *(double *)a3.m128_u64, a4, a5);
      return sub_15FB440(25, v47, v43, (__int64)v65, 0);
    }
    v18 = *(_QWORD *)(v14 + 8);
    v19 = *(_BYTE *)(v14 + 16);
    if ( !v18 || *(_QWORD *)(v18 + 8) )
      goto LABEL_11;
    if ( v19 > 0x17u )
    {
      if ( v19 != 60 )
      {
LABEL_11:
        if ( v19 == 75 )
          return sub_174DF90(a1, v14, (__int64 ***)a2, a3, a4, a5, a6, v16, v17, a9, a10);
LABEL_12:
        v60 = 0;
        v65[0] = &v60;
        v65[1] = &v61;
        v61 = 0;
        v62 = 0;
        v66 = &v62;
        if ( (unsigned __int8)sub_174BC40(v65, v14) )
        {
          v20 = v62;
          if ( v61 == v62 && *(_QWORD *)a2 == *v60 )
          {
            v59 = *(__int64 ***)a2;
            v48 = sub_16431D0(*(_QWORD *)v14);
            v49 = sub_16431D0((__int64)v59);
            v50 = *(_QWORD **)(v20 + 24);
            if ( *(_DWORD *)(v20 + 32) > 0x40u )
              v50 = (_QWORD *)*v50;
            v51 = sub_15A0680((__int64)v59, (unsigned int)(v49 - v48 + (_DWORD)v50), 0);
            v52 = a1[1];
            v53 = v51;
            v54 = sub_1649960(a2);
            LOWORD(v66) = 261;
            v63[0] = (__int64)v54;
            v63[1] = v55;
            v65[0] = v63;
            v56 = (__int64 *)sub_173DC60(v52, (__int64)v60, v53, (__int64 *)v65, 0, 0, *(double *)a3.m128_u64, a4, a5);
            LOWORD(v66) = 257;
            v60 = v56;
            return sub_15FB440(25, v56, v53, (__int64)v65, 0);
          }
        }
        if ( LOBYTE(qword_4FA1CA0[20]) )
        {
          v21 = *(_BYTE *)(v14 + 16);
          if ( v21 == 35 )
          {
            v22 = *(_QWORD *)(v14 - 48);
            if ( !v22 )
              return 0;
            v23 = *(_QWORD *)(v14 - 24);
            if ( *(_BYTE *)(v23 + 16) != 13 )
              return 0;
          }
          else
          {
            if ( v21 != 5 )
              return 0;
            if ( *(_WORD *)(v14 + 18) != 11 )
              return 0;
            v22 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
            if ( !v22 )
              return 0;
            v23 = *(_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
            if ( *(_BYTE *)(v23 + 16) != 13 )
              return 0;
          }
          v24 = a1[1];
          v25 = *(__int64 ***)a2;
          LOWORD(v66) = 257;
          v26 = sub_1708970(v24, 38, v22, v25, (__int64 *)v65);
          v27 = a1[1];
          v28 = *(__int64 ***)a2;
          v29 = (__int64 *)v26;
          LOWORD(v66) = 257;
          v30 = sub_1708970(v27, 38, v23, v28, (__int64 *)v65);
          LOWORD(v66) = 257;
          return sub_15FB440(11, v29, (__int64)v30, (__int64)v65, 0);
        }
        return 0;
      }
    }
    else if ( v19 != 5 || *(_WORD *)(v14 + 18) != 36 )
    {
      goto LABEL_12;
    }
    if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
      v38 = *(__int64 *****)(v14 - 8);
    else
      v38 = (__int64 ****)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
    if ( *v38 && v13 == **v38 )
    {
      v58 = *v38;
      v39 = sub_16431D0(v15);
      v40 = sub_16431D0((__int64)v13);
      v41 = sub_15A0680((__int64)v13, (unsigned int)(v40 - v39), 0);
      LOWORD(v66) = 257;
      v42 = a1[1];
      v43 = v41;
      v64 = 257;
      v44 = v41;
      v45 = (__int64)v58;
      goto LABEL_39;
    }
    goto LABEL_11;
  }
  return result;
}
