// Function: sub_33F2320
// Address: 0x33f2320
//
_QWORD *__fastcall sub_33F2320(_QWORD *a1, __int64 a2, unsigned __int64 a3, int *a4)
{
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 v11; // r8
  __int16 *v12; // rcx
  _QWORD *v13; // rbx
  __int64 v14; // r15
  unsigned __int16 v15; // dx
  int v16; // eax
  unsigned __int64 v17; // rax
  _QWORD *result; // rax
  __int16 v19; // ax
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rsi
  unsigned __int16 *v23; // rcx
  unsigned __int16 v24; // r12
  __int64 v25; // rax
  __int16 *v26; // rcx
  int v27; // eax
  bool v28; // al
  unsigned int v29; // eax
  unsigned int v30; // ecx
  __int64 v31; // r8
  unsigned __int64 v32; // r14
  unsigned __int64 v33; // r14
  int v35; // eax
  bool v37; // al
  __int16 *v38; // rcx
  int *v39; // rdx
  __int64 v40; // rax
  int v41; // r12d
  unsigned __int16 *v42; // rcx
  unsigned __int16 v43; // r14
  __int64 v44; // rax
  int v45; // ecx
  unsigned int v46; // eax
  _QWORD *v47; // [rsp+8h] [rbp-88h]
  __int64 v48; // [rsp+8h] [rbp-88h]
  __int64 v49; // [rsp+8h] [rbp-88h]
  unsigned __int16 *v50; // [rsp+8h] [rbp-88h]
  __int16 *v51; // [rsp+8h] [rbp-88h]
  __int64 v52; // [rsp+10h] [rbp-80h] BYREF
  __int64 v53; // [rsp+18h] [rbp-78h]
  unsigned __int64 v54; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v55; // [rsp+28h] [rbp-68h]
  unsigned __int64 v56; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v57; // [rsp+38h] [rbp-58h]
  unsigned __int64 v58; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v59; // [rsp+48h] [rbp-48h]
  __int64 v60; // [rsp+50h] [rbp-40h] BYREF
  __int64 v61; // [rsp+58h] [rbp-38h]

  v9 = sub_33CF690(a2, a3);
  v11 = v10;
  v12 = *(__int16 **)(v9 + 48);
  v13 = (_QWORD *)v9;
  v14 = 8LL * v10;
  v15 = v12[v14];
  v53 = *(_QWORD *)&v12[v14 + 4];
  v16 = *(_DWORD *)(v9 + 24);
  LOWORD(v52) = v15;
  if ( v16 != 165 )
  {
    if ( v16 == 168 )
    {
      *a4 = 0;
      return v13;
    }
    v55 = 1;
    v54 = 0;
    if ( v15 )
    {
      if ( (unsigned __int16)(v15 - 176) <= 0x34u )
      {
LABEL_5:
        v57 = 1;
        v17 = 1;
        goto LABEL_6;
      }
      v29 = word_4456340[v15 - 1];
    }
    else
    {
      v48 = v11;
      v28 = sub_3007100((__int64)&v52);
      v11 = v48;
      if ( v28 )
        goto LABEL_5;
      v29 = sub_3007130((__int64)&v52, a3);
      v11 = v48;
    }
    v57 = v29;
    if ( v29 > 0x40 )
    {
      v49 = v11;
      sub_C43690((__int64)&v56, -1, 1);
      v11 = v49;
      goto LABEL_7;
    }
    if ( v29 )
      v17 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v29;
    else
      v17 = 0;
LABEL_6:
    v56 = v17;
LABEL_7:
    if ( (unsigned __int8)sub_33CD9D0(
                            (__int64)a1,
                            (__int64)v13,
                            v11 | a3 & 0xFFFFFFFF00000000LL,
                            (__int64)&v56,
                            (__int64)&v54,
                            0) )
    {
      if ( (_WORD)v52 )
      {
        if ( (unsigned __int16)(v52 - 176) <= 0x34u )
        {
LABEL_10:
          *a4 = 0;
LABEL_11:
          result = v13;
          if ( v57 > 0x40 && v56 )
          {
            j_j___libc_free_0_0(v56);
            result = v13;
          }
          if ( v55 > 0x40 )
          {
            if ( v54 )
            {
              v47 = result;
              j_j___libc_free_0_0(v54);
              return v47;
            }
          }
          return result;
        }
      }
      else if ( sub_3007100((__int64)&v52) )
      {
        goto LABEL_10;
      }
      if ( v57 <= 0x40 )
      {
        if ( (v56 & ~v54) == 0 )
        {
LABEL_36:
          v30 = v52;
          v31 = v53;
          *a4 = 0;
          v60 = 0;
          LODWORD(v61) = 0;
          v13 = sub_33F17F0(a1, 51, (__int64)&v60, v30, v31);
          if ( v60 )
            sub_B91220((__int64)&v60, v60);
          goto LABEL_11;
        }
      }
      else if ( (unsigned __int8)sub_C446F0((__int64 *)&v56, (__int64 *)&v54) )
      {
        goto LABEL_36;
      }
      v59 = v55;
      if ( v55 > 0x40 )
      {
        sub_C43780((__int64)&v58, (const void **)&v54);
        if ( v59 > 0x40 )
        {
          sub_C43B90(&v58, (__int64 *)&v56);
          v46 = v59;
          v33 = v58;
          v59 = 0;
          LODWORD(v61) = v46;
          v60 = v58;
          if ( v46 > 0x40 )
          {
            *a4 = sub_C445E0((__int64)&v60);
            if ( v33 )
            {
              j_j___libc_free_0_0(v33);
              if ( v59 > 0x40 )
              {
                if ( v58 )
                  j_j___libc_free_0_0(v58);
              }
            }
            goto LABEL_11;
          }
          goto LABEL_42;
        }
        v32 = v58;
      }
      else
      {
        v32 = v54;
      }
      v33 = v56 & v32;
LABEL_42:
      _R14 = ~v33;
      v35 = 64;
      __asm { tzcnt   rdx, r14 }
      if ( _R14 )
        v35 = _RDX;
      *a4 = v35;
      goto LABEL_11;
    }
    if ( v57 > 0x40 && v56 )
      j_j___libc_free_0_0(v56);
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    return 0;
  }
  v19 = *v12;
  v20 = *((_QWORD *)v12 + 1);
  LOWORD(v60) = v19;
  v61 = v20;
  if ( v19 )
  {
    if ( (unsigned __int16)(v19 - 176) > 0x34u )
      goto LABEL_24;
  }
  else if ( !sub_3007100((__int64)&v60) )
  {
    goto LABEL_19;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v60 )
  {
    if ( (unsigned __int16)(v60 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_24:
    v22 = word_4456340[(unsigned __int16)v60 - 1];
    if ( !(unsigned __int8)sub_33E2340(v13[12], v22) )
      return 0;
    goto LABEL_25;
  }
LABEL_19:
  v21 = sub_3007130((__int64)&v60, a3);
  v22 = v21;
  if ( !(unsigned __int8)sub_33E2340(v13[12], v21) )
    return 0;
LABEL_25:
  v23 = (unsigned __int16 *)v13[6];
  v24 = *v23;
  v25 = *((_QWORD *)v23 + 1);
  LOWORD(v60) = v24;
  v61 = v25;
  if ( v24 )
  {
    if ( (unsigned __int16)(v24 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v26 = (__int16 *)v13[6];
    v27 = word_4456340[v24 - 1];
  }
  else
  {
    v50 = v23;
    v37 = sub_3007100((__int64)&v60);
    v38 = (__int16 *)v50;
    if ( v37 )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      v38 = (__int16 *)v13[6];
    }
    v51 = v38;
    v27 = sub_3007130((__int64)&v60, v22);
    v26 = v51;
  }
  v39 = (int *)v13[12];
  if ( v27 )
  {
    v40 = (__int64)&v39[v27 - 1 + 1];
    while ( 1 )
    {
      v41 = *v39;
      if ( *v39 >= 0 )
        break;
      if ( (int *)v40 == ++v39 )
        goto LABEL_65;
    }
  }
  else
  {
LABEL_65:
    v41 = 0;
  }
  v42 = (unsigned __int16 *)&v26[v14];
  v43 = *v42;
  v44 = *((_QWORD *)v42 + 1);
  LOWORD(v60) = v43;
  v61 = v44;
  if ( v43 )
  {
    if ( (unsigned __int16)(v43 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v45 = word_4456340[v43 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v60) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v45 = sub_3007130((__int64)&v60, v22);
  }
  *a4 = v41 % v45;
  return *(_QWORD **)(v13[5] + 40LL * (unsigned int)(v41 / v45));
}
