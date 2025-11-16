// Function: sub_33DE230
// Address: 0x33de230
//
__int64 __fastcall sub_33DE230(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned int a6)
{
  char *v6; // r15
  int v8; // edx
  _QWORD **v11; // r10
  unsigned __int8 v13; // r11
  char v14; // al
  _QWORD **v15; // r10
  unsigned int v16; // r11d
  char *v17; // r13
  __int64 v18; // rax
  char *v19; // r12
  __int64 v20; // rax
  unsigned int v21; // r14d
  char v22; // al
  unsigned __int16 *v23; // rdx
  unsigned __int16 v24; // ax
  __int64 v25; // rdx
  unsigned int v26; // eax
  unsigned int v27; // r13d
  _QWORD **v28; // r10
  unsigned __int8 v29; // r11
  int v30; // eax
  int v31; // eax
  unsigned int v32; // r14d
  __int64 v33; // rax
  unsigned __int8 v34; // r9
  __int64 v35; // r14
  __int64 v36; // rax
  char v37; // al
  bool v38; // al
  char v39; // al
  unsigned int v40; // r14d
  signed __int64 v41; // rax
  unsigned __int8 v42; // [rsp+Ch] [rbp-74h]
  unsigned __int8 v43; // [rsp+Ch] [rbp-74h]
  unsigned int v44; // [rsp+10h] [rbp-70h]
  unsigned int v45; // [rsp+10h] [rbp-70h]
  _QWORD **v46; // [rsp+10h] [rbp-70h]
  unsigned __int8 v47; // [rsp+10h] [rbp-70h]
  _QWORD **v48; // [rsp+10h] [rbp-70h]
  unsigned __int8 v49; // [rsp+10h] [rbp-70h]
  unsigned __int8 v50; // [rsp+10h] [rbp-70h]
  unsigned __int8 v51; // [rsp+10h] [rbp-70h]
  unsigned __int8 v52; // [rsp+10h] [rbp-70h]
  unsigned int v53; // [rsp+10h] [rbp-70h]
  unsigned int v54; // [rsp+10h] [rbp-70h]
  _QWORD **v55; // [rsp+18h] [rbp-68h]
  _QWORD **v56; // [rsp+18h] [rbp-68h]
  __int64 v57; // [rsp+18h] [rbp-68h]
  _QWORD **v58; // [rsp+18h] [rbp-68h]
  _QWORD **v59; // [rsp+18h] [rbp-68h]
  _QWORD **v60; // [rsp+18h] [rbp-68h]
  _QWORD **v61; // [rsp+18h] [rbp-68h]
  _QWORD **v62; // [rsp+18h] [rbp-68h]
  unsigned __int64 v63; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v64; // [rsp+28h] [rbp-58h]
  unsigned __int64 v65; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v66; // [rsp+38h] [rbp-48h]
  unsigned __int16 v67; // [rsp+40h] [rbp-40h] BYREF
  __int64 v68; // [rsp+48h] [rbp-38h]

  v8 = *(_DWORD *)(a2 + 24);
  if ( v8 == 52 )
    goto LABEL_4;
  if ( a6 > 5 )
    goto LABEL_6;
  LOBYTE(v6) = (unsigned int)(v8 - 11) <= 1 || (unsigned int)(v8 - 35) <= 1;
  if ( (_BYTE)v6 )
  {
LABEL_4:
    LODWORD(v6) = 1;
    return (unsigned int)v6;
  }
  v11 = a1;
  if ( v8 == 156 )
  {
    v32 = *(_DWORD *)(a2 + 64);
    if ( !v32 )
      goto LABEL_4;
    v33 = v32;
    v34 = a5;
    v35 = 0;
    v57 = v33;
    while ( 1 )
    {
      v36 = *(_QWORD *)a4;
      if ( *(_DWORD *)(a4 + 8) > 0x40u )
        v36 = *(_QWORD *)(v36 + 8LL * ((unsigned int)v35 >> 6));
      if ( (v36 & (1LL << v35)) != 0 )
      {
        v43 = v34;
        v48 = v11;
        v37 = sub_33DE850(
                v11,
                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40 * v35),
                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40 * v35 + 8),
                v34,
                a6 + 1);
        v11 = v48;
        v34 = v43;
        if ( !v37 )
          break;
      }
      if ( v57 == ++v35 )
        goto LABEL_4;
    }
LABEL_6:
    LODWORD(v6) = 0;
    return (unsigned int)v6;
  }
  v13 = a5;
  if ( v8 <= 156 )
  {
    switch ( v8 )
    {
      case 7:
      case 8:
      case 15:
      case 39:
      case 50:
        goto LABEL_4;
      case 51:
        LODWORD(v6) = a5;
        break;
      default:
        goto LABEL_13;
    }
    return (unsigned int)v6;
  }
  if ( v8 == 165 )
  {
    v23 = *(unsigned __int16 **)(a2 + 48);
    v64 = 1;
    v66 = 1;
    v63 = 0;
    v65 = 0;
    v24 = *v23;
    v25 = *((_QWORD *)v23 + 1);
    v67 = v24;
    v68 = v25;
    if ( v24 )
    {
      if ( (unsigned __int16)(v24 - 176) > 0x34u )
      {
LABEL_26:
        v26 = word_4456340[v67 - 1];
        goto LABEL_27;
      }
    }
    else
    {
      v49 = a5;
      v38 = sub_3007100((__int64)&v67);
      v11 = a1;
      v13 = v49;
      if ( !v38 )
        goto LABEL_48;
    }
    v51 = v13;
    v59 = v11;
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    v11 = v59;
    v13 = v51;
    if ( v67 )
    {
      if ( (unsigned __int16)(v67 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
        v13 = v51;
        v11 = v59;
      }
      goto LABEL_26;
    }
LABEL_48:
    v50 = v13;
    v58 = v11;
    v26 = sub_3007130((__int64)&v67, a2);
    v13 = v50;
    v11 = v58;
LABEL_27:
    v42 = v13;
    v46 = v11;
    if ( !(unsigned __int8)sub_9B7FB0(
                             *(_DWORD *)(a4 + 8),
                             *(_DWORD **)(a2 + 96),
                             v26,
                             (__int64 *)a4,
                             (__int64 *)&v63,
                             (__int64 *)&v65,
                             0) )
      goto LABEL_32;
    v27 = v64;
    v28 = v46;
    v29 = v42;
    if ( v64 <= 0x40 )
    {
      if ( !v63 )
        goto LABEL_30;
    }
    else
    {
      v30 = sub_C444A0((__int64)&v63);
      v28 = v46;
      v29 = v42;
      if ( v27 == v30 )
        goto LABEL_30;
    }
    v52 = v29;
    v60 = v28;
    v39 = sub_33DE230(v28, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), &v63, v29, a6 + 1);
    v28 = v60;
    v29 = v52;
    if ( !v39 )
    {
LABEL_32:
      if ( v66 > 0x40 && v65 )
        j_j___libc_free_0_0(v65);
      if ( v64 > 0x40 )
      {
        if ( v63 )
          j_j___libc_free_0_0(v63);
      }
      return (unsigned int)v6;
    }
LABEL_30:
    v47 = v29;
    v56 = v28;
    LOBYTE(v31) = sub_9867B0((__int64)&v65);
    LODWORD(v6) = v31;
    if ( !(_BYTE)v31 )
      LODWORD(v6) = sub_33DE230(
                      v56,
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                      &v65,
                      v47,
                      a6 + 1);
    goto LABEL_32;
  }
  if ( v8 != 168 )
  {
LABEL_13:
    LOBYTE(v6) = (unsigned int)v8 > 0x1F3 || (unsigned int)(v8 - 46) <= 2;
    if ( (_BYTE)v6 )
    {
      LODWORD(v6) = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD **, _QWORD, _QWORD))(*a1[2] + 2080LL))(
                      a1[2],
                      a2,
                      a3,
                      a4,
                      a1,
                      (unsigned __int8)a5,
                      a6);
      return (unsigned int)v6;
    }
    v44 = (unsigned __int8)a5;
    v14 = sub_33DE0A0(a1, a2, a3, a5, 1u, a6);
    v15 = a1;
    v16 = v44;
    if ( v14 )
      return (unsigned int)v6;
    v17 = *(char **)(a2 + 40);
    v18 = 40LL * *(unsigned int *)(a2 + 64);
    v19 = &v17[v18];
    v20 = (__int64)(0xCCCCCCCCCCCCCCCDLL * (v18 >> 3)) >> 2;
    if ( v20 )
    {
      v21 = a6 + 1;
      v6 = &v17[160 * v20];
      do
      {
        v45 = v16;
        v55 = v15;
        if ( !(unsigned __int8)sub_33DE850(v15, *(_QWORD *)v17, *((_QWORD *)v17 + 1), v16, v21) )
          goto LABEL_22;
        if ( !(unsigned __int8)sub_33DE850(v55, *((_QWORD *)v17 + 5), *((_QWORD *)v17 + 6), v45, v21) )
        {
          v17 += 40;
          goto LABEL_22;
        }
        if ( !(unsigned __int8)sub_33DE850(v55, *((_QWORD *)v17 + 10), *((_QWORD *)v17 + 11), v45, v21) )
        {
          v17 += 80;
          goto LABEL_22;
        }
        v22 = sub_33DE850(v55, *((_QWORD *)v17 + 15), *((_QWORD *)v17 + 16), v45, v21);
        v15 = v55;
        v16 = v45;
        if ( !v22 )
        {
          v17 += 120;
          goto LABEL_22;
        }
        v17 += 160;
      }
      while ( v17 != v6 );
    }
    v40 = a6 + 1;
    v41 = v19 - v17;
    if ( v19 - v17 != 80 )
    {
      if ( v41 != 120 )
      {
        v40 = a6 + 1;
        if ( v41 != 40 )
        {
          v17 = v19;
LABEL_22:
          LOBYTE(v6) = v19 == v17;
          return (unsigned int)v6;
        }
LABEL_67:
        LODWORD(v6) = sub_33DE850(v15, *(_QWORD *)v17, *((_QWORD *)v17 + 1), v16, v40);
        if ( (_BYTE)v6 )
          return (unsigned int)v6;
        goto LABEL_22;
      }
      v40 = a6 + 1;
      v53 = v16;
      v61 = v15;
      if ( !(unsigned __int8)sub_33DE850(v15, *(_QWORD *)v17, *((_QWORD *)v17 + 1), v16, a6 + 1) )
        goto LABEL_22;
      v16 = v53;
      v15 = v61;
      v17 += 40;
    }
    v54 = v16;
    v62 = v15;
    if ( !(unsigned __int8)sub_33DE850(v15, *(_QWORD *)v17, *((_QWORD *)v17 + 1), v16, v40) )
      goto LABEL_22;
    v16 = v54;
    v15 = v62;
    v17 += 40;
    goto LABEL_67;
  }
  return sub_33DE850(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (unsigned __int8)a5, a6 + 1);
}
