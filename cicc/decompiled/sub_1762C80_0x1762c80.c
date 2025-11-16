// Function: sub_1762C80
// Address: 0x1762c80
//
_QWORD *__fastcall sub_1762C80(
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
  __int64 v10; // r12
  _BYTE *v11; // rbx
  __int64 ***v12; // r14
  char v13; // r13
  __int64 **v14; // r15
  bool v15; // al
  unsigned __int64 v16; // r11
  bool v17; // r10
  unsigned __int8 v18; // al
  __int64 ***v19; // rax
  __int64 **v20; // rdx
  __int64 v21; // r15
  const char *v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // rdi
  double v25; // xmm4_8
  double v26; // xmm5_8
  int v27; // eax
  _QWORD *result; // rax
  __int64 v29; // rsi
  __int64 v30; // r13
  unsigned __int64 v31; // rdi
  unsigned __int8 v32; // al
  __int64 ***v33; // r13
  __int64 v34; // rcx
  __int64 ****v35; // rdi
  __int64 **v36; // rsi
  __int64 **v37; // rax
  __int64 **v38; // rcx
  int v39; // edx
  __int64 **v40; // rax
  __int64 ***v41; // rdx
  __int64 v42; // [rsp+0h] [rbp-80h]
  bool v43; // [rsp+Eh] [rbp-72h]
  bool v44; // [rsp+Fh] [rbp-71h]
  __int64 **v46; // [rsp+18h] [rbp-68h]
  __int64 ***v47; // [rsp+18h] [rbp-68h]
  __int64 v48; // [rsp+18h] [rbp-68h]
  _QWORD *v49; // [rsp+18h] [rbp-68h]
  _QWORD v50[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v51[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v52; // [rsp+40h] [rbp-40h]

  v10 = a2;
  v11 = *(_BYTE **)(a2 - 48);
  v12 = (__int64 ***)*((_QWORD *)v11 - 3);
  v13 = v11[16];
  v46 = *(__int64 ***)v11;
  v14 = *v12;
  if ( v13 == 69 )
  {
    v29 = (__int64)*v12;
    v30 = *(_QWORD *)v11;
    if ( *((_BYTE *)v14 + 8) == 16 )
    {
      v29 = (__int64)v14[3];
      v30 = *(_QWORD *)(*(_QWORD *)v11 + 24LL);
    }
    if ( (unsigned int)sub_15A9570(a1[333], v29) != *(_DWORD *)(v30 + 8) >> 8 )
      goto LABEL_19;
    v31 = *(_QWORD *)(v10 - 24);
    v32 = *(_BYTE *)(v31 + 16);
    if ( v32 > 0x17u )
    {
      if ( v32 != 69 )
      {
LABEL_19:
        v13 = v11[16];
        goto LABEL_2;
      }
      goto LABEL_28;
    }
    if ( v32 == 5 )
    {
      if ( *(_WORD *)(v31 + 18) == 45 )
      {
LABEL_28:
        if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
          v35 = *(__int64 *****)(v31 - 8);
        else
          v35 = (__int64 ****)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
        v33 = *v35;
        v36 = **v35;
        v37 = v36;
        if ( *((_BYTE *)v36 + 8) == 16 )
          v37 = (__int64 **)*v36[2];
        v38 = *v12;
        v39 = *((_DWORD *)v37 + 2) >> 8;
        v40 = *v12;
        if ( *((_BYTE *)*v12 + 8) == 16 )
          v40 = (__int64 **)*v38[2];
        if ( v39 != *((_DWORD *)v40 + 2) >> 8 )
          goto LABEL_19;
        if ( v36 == v38 )
          goto LABEL_47;
        v41 = *v35;
        v52 = 257;
        v33 = (__int64 ***)sub_1708970(a1[1], 47, (__int64)v41, *v12, v51);
LABEL_46:
        if ( !v33 )
          goto LABEL_19;
LABEL_47:
        LOWORD(v10) = *(_WORD *)(v10 + 18) & 0x7FFF;
LABEL_26:
        v52 = 257;
        result = sub_1648A60(56, 2u);
        if ( !result )
          return result;
        v34 = (__int64)v33;
LABEL_42:
        v49 = result;
        sub_17582E0((__int64)result, v10, (__int64)v12, v34, (__int64)v51);
        return v49;
      }
    }
    else if ( v32 > 0x10u )
    {
      goto LABEL_19;
    }
    v33 = (__int64 ***)sub_15A3BA0(v31, v14, 0);
    goto LABEL_46;
  }
LABEL_2:
  if ( (unsigned __int8)(v13 - 61) > 1u )
    return 0;
  v44 = v13 == 62;
  v15 = sub_15FF7F0(*(_WORD *)(v10 + 18) & 0x7FFF);
  v16 = *(_QWORD *)(v10 - 24);
  v17 = v15;
  v18 = *(_BYTE *)(v16 + 16);
  if ( v18 > 0x17u )
  {
    if ( (unsigned int)v18 - 60 > 0xC )
      return 0;
    v33 = *(__int64 ****)(v16 - 24);
    if ( *v12 != *v33 || v18 != v11[16] )
      return 0;
    LODWORD(v10) = *(_WORD *)(v10 + 18) & 0x7FFF;
    if ( (unsigned int)(v10 - 32) > 1 && (!v44 || !v17) )
      LOWORD(v10) = sub_15FF470(v10);
    goto LABEL_26;
  }
  v43 = v17;
  if ( v18 > 0x10u )
    return 0;
  v42 = *(_QWORD *)(v10 - 24);
  v19 = (__int64 ***)sub_15A43B0(v16, v14, 0);
  v20 = v46;
  v47 = v19;
  if ( v42 != sub_15A46C0((unsigned int)(unsigned __int8)v11[16] - 24, v19, v20, 0) )
  {
    if ( v13 == 62 && !v43 && *(_BYTE *)(v42 + 16) == 13 )
    {
      v21 = sub_15A04A0(v14);
      v48 = a1[1];
      v22 = sub_1649960(v10);
      v50[1] = v23;
      v50[0] = v22;
      v52 = 261;
      v51[0] = (__int64)v50;
      v24 = (__int64 *)sub_17203D0(v48, 38, (__int64)v12, v21, v51);
      v27 = *(unsigned __int16 *)(v10 + 18);
      BYTE1(v27) &= ~0x80u;
      if ( v27 == 36 )
        return (_QWORD *)sub_170E100(a1, v10, (__int64)v24, a3, a4, a5, a6, v25, v26, a9, a10);
      v52 = 257;
      return (_QWORD *)sub_15FB630(v24, (__int64)v51, 0);
    }
    return 0;
  }
  LODWORD(v10) = *(_WORD *)(v10 + 18) & 0x7FFF;
  if ( (unsigned int)(v10 - 32) > 1 && (v13 != 62 || !v43) )
    LOWORD(v10) = sub_15FF470(v10);
  v52 = 257;
  result = sub_1648A60(56, 2u);
  if ( result )
  {
    v34 = (__int64)v47;
    goto LABEL_42;
  }
  return result;
}
