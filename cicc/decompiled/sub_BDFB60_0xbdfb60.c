// Function: sub_BDFB60
// Address: 0xbdfb60
//
char __fastcall sub_BDFB60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  const char *v7; // r13
  int v8; // ecx
  __int64 v9; // rdx
  _BYTE *v10; // r15
  int v11; // ecx
  const char *v12; // rax
  __int64 v13; // r13
  _BYTE *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  const char *v17; // rsi
  int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r8
  const char *v22; // r13
  unsigned __int8 **v23; // rax
  unsigned __int8 *v24; // rax
  _BYTE *v25; // r11
  unsigned __int8 **v26; // rax
  _BYTE *v27; // rax
  __int64 v28; // rax
  unsigned __int8 v29; // dl
  __int64 v30; // r13
  unsigned __int64 v31; // r15
  unsigned __int64 v32; // rax
  const char *v33; // r15
  int v34; // edx
  __int64 v35; // rax
  _BYTE **v36; // rdx
  _BYTE **v37; // r15
  _BYTE *v38; // r13
  __int64 v39; // rax
  unsigned __int8 **v40; // rax
  unsigned __int8 **v41; // rax
  unsigned __int8 *v42; // r15
  __int64 v43; // rcx
  __int64 v44; // r8
  const char *v45; // r12
  unsigned __int64 v46; // rbx
  _BYTE *v48; // [rsp+8h] [rbp-F8h]
  __int64 v49; // [rsp+10h] [rbp-F0h]
  char *v50; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v51; // [rsp+18h] [rbp-E8h]
  _BYTE *v52; // [rsp+20h] [rbp-E0h]
  _BYTE *v53; // [rsp+28h] [rbp-D8h]
  _BYTE **v54; // [rsp+28h] [rbp-D8h]
  const char *v55; // [rsp+30h] [rbp-D0h]
  __int64 v56; // [rsp+30h] [rbp-D0h]
  __int64 v57; // [rsp+38h] [rbp-C8h]
  unsigned __int8 *v58; // [rsp+38h] [rbp-C8h]
  __int64 v59; // [rsp+38h] [rbp-C8h]
  __int64 v60; // [rsp+38h] [rbp-C8h]
  __m128i v61; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v62; // [rsp+50h] [rbp-B0h]
  __int64 v63; // [rsp+58h] [rbp-A8h]
  __int16 v64; // [rsp+60h] [rbp-A0h]
  __m128i v65; // [rsp+70h] [rbp-90h] BYREF
  __int64 v66; // [rsp+80h] [rbp-80h]
  __int64 v67; // [rsp+88h] [rbp-78h]
  __int16 v68; // [rsp+90h] [rbp-70h]
  __m128i v69; // [rsp+A0h] [rbp-60h] BYREF
  const char *v70; // [rsp+B0h] [rbp-50h]
  __int16 v71; // [rsp+C0h] [rbp-40h]

  v6 = *(_DWORD *)(a4 + 4) & 0x7FFFFFF;
  v7 = *(const char **)(*(_QWORD *)(a4 - 32 * v6) + 24LL);
  v8 = *(unsigned __int8 *)v7;
  if ( (unsigned int)(v8 - 1) > 1 && (_BYTE)v8 != 4 )
  {
    if ( (unsigned __int8)(v8 - 5) > 0x1Fu
      || ((*(v7 - 16) & 2) == 0 ? (v18 = (*((_WORD *)v7 - 8) >> 6) & 0xF) : (v18 = *((_DWORD *)v7 - 6)), v18) )
    {
      v66 = a2;
      v65.m128i_i64[0] = (__int64)"invalid llvm.dbg.";
      v69.m128i_i64[0] = (__int64)&v65;
      v68 = 1283;
      v67 = a3;
      v70 = " intrinsic address/value";
      v71 = 770;
      goto LABEL_25;
    }
  }
  v7 = *(const char **)(*(_QWORD *)(a4 + 32 * (1 - v6)) + 24LL);
  if ( *v7 != 26 )
  {
    v66 = a2;
    v65.m128i_i64[0] = (__int64)"invalid llvm.dbg.";
    v69.m128i_i64[0] = (__int64)&v65;
    v68 = 1283;
    v67 = a3;
    v70 = " intrinsic variable";
    v71 = 770;
    goto LABEL_25;
  }
  v7 = *(const char **)(*(_QWORD *)(a4 + 32 * (2 - v6)) + 24LL);
  if ( *v7 != 7 )
  {
    v66 = a2;
    v65.m128i_i64[0] = (__int64)"invalid llvm.dbg.";
    v69.m128i_i64[0] = (__int64)&v65;
    v68 = 1283;
    v67 = a3;
    v70 = " intrinsic expression";
    v71 = 770;
    goto LABEL_25;
  }
  v9 = *(_QWORD *)(a4 - 32);
  if ( !v9 || *(_BYTE *)v9 || *(_QWORD *)(v9 + 24) != *(_QWORD *)(a4 + 80) )
    BUG();
  if ( *(_DWORD *)(v9 + 36) != 68 )
    goto LABEL_30;
  v7 = *(const char **)(*(_QWORD *)(a4 + 32 * (3 - v6)) + 24LL);
  if ( *v7 != 30 )
  {
    v69.m128i_i64[0] = (__int64)"invalid llvm.dbg.assign intrinsic DIAssignID";
    v71 = 259;
LABEL_25:
    LOBYTE(v16) = sub_BDD6D0((__int64 *)a1, (__int64)&v69);
    if ( *(_QWORD *)a1 )
    {
      sub_BDBD80(a1, (_BYTE *)a4);
      LOBYTE(v16) = (unsigned __int8)sub_BD9900((__int64 *)a1, v7);
    }
    return v16;
  }
  v10 = *(_BYTE **)(*(_QWORD *)(a4 + 32 * (4 - v6)) + 24LL);
  v11 = (unsigned __int8)*v10;
  if ( (unsigned int)(v11 - 1) <= 1
    || (unsigned __int8)(v11 - 5) <= 0x1Fu
    && ((*(v10 - 16) & 2) == 0 ? (v34 = (*((_WORD *)v10 - 8) >> 6) & 0xF) : (v34 = *((_DWORD *)v10 - 6)), !v34) )
  {
    v10 = *(_BYTE **)(*(_QWORD *)(a4 + 32 * (5 - v6)) + 24LL);
    if ( *v10 != 7 )
    {
      HIBYTE(v71) = 1;
      v12 = "invalid llvm.dbg.assign intrinsic address expression";
      goto LABEL_13;
    }
    v59 = a3;
    v35 = sub_AE9410((__int64)v7);
    a3 = v59;
    v54 = v36;
    v37 = (_BYTE **)v35;
    if ( v36 != (_BYTE **)v35 )
    {
      while ( 1 )
      {
        v38 = *v37;
        v56 = a3;
        v60 = sub_B43CB0(a4);
        v39 = sub_B43CB0((__int64)v38);
        a3 = v56;
        if ( v60 != v39 )
          break;
        if ( v54 == ++v37 )
          goto LABEL_30;
      }
      v69.m128i_i64[0] = (__int64)"inst not in same function as dbg.assign";
      v71 = 259;
      LOBYTE(v16) = sub_BDD6D0((__int64 *)a1, (__int64)&v69);
      if ( *(_QWORD *)a1 )
      {
        if ( v38 )
          sub_BDBD80(a1, v38);
        LOBYTE(v16) = (unsigned __int8)sub_BDBD80(a1, (_BYTE *)a4);
      }
      return v16;
    }
LABEL_30:
    v16 = *(_QWORD *)(a4 + 48);
    if ( v16 && *(_BYTE *)v16 != 6 )
      return v16;
    v53 = 0;
    v19 = *(_QWORD *)(a4 + 40);
    v48 = (_BYTE *)v19;
    if ( v19 )
      v53 = *(_BYTE **)(v19 + 72);
    v57 = a3;
    v55 = *(const char **)(*(_QWORD *)(a4 + 32 * (1LL - (*(_DWORD *)(a4 + 4) & 0x7FFFFFF))) + 24LL);
    v22 = (const char *)sub_B10CD0(a4 + 48);
    if ( !v22 )
    {
      v63 = v57;
      v65.m128i_i64[0] = (__int64)" intrinsic requires a !dbg attachment";
      v64 = 1283;
      v61.m128i_i64[0] = (__int64)"llvm.dbg.";
      v68 = 259;
      v62 = a2;
      sub_9C6370(&v69, &v61, &v65, v20, v21, v57);
      LOBYTE(v16) = sub_BDD6D0((__int64 *)a1, (__int64)&v69);
      if ( *(_QWORD *)a1 )
      {
        sub_BDBD80(a1, (_BYTE *)a4);
        LOBYTE(v16) = (_BYTE)v48;
        if ( v48 )
          LOBYTE(v16) = (unsigned __int8)sub_BDBD80(a1, v48);
        if ( v53 )
          LOBYTE(v16) = (unsigned __int8)sub_BDBD80(a1, v53);
      }
      return v16;
    }
    v49 = v57;
    v50 = (char *)(v55 - 16);
    v23 = (unsigned __int8 **)sub_A17150((_BYTE *)v55 - 16);
    v24 = sub_BDA280(*v23);
    v52 = v25;
    v58 = v24;
    v26 = (unsigned __int8 **)sub_A17150(v25);
    v16 = (__int64)sub_BDA280(*v26);
    if ( !v58 || !v16 )
      return v16;
    if ( v58 != (unsigned __int8 *)v16 )
    {
      v40 = (unsigned __int8 **)sub_A17150(v52);
      v51 = sub_AF34D0(*v40);
      v41 = (unsigned __int8 **)sub_A17150((_BYTE *)v55 - 16);
      v42 = sub_AF34D0(*v41);
      v62 = a2;
      v68 = 259;
      v65.m128i_i64[0] = (__int64)" variable and !dbg attachment";
      v64 = 1283;
      v61.m128i_i64[0] = (__int64)"mismatched subprogram between llvm.dbg.";
      v63 = v49;
      sub_9C6370(&v69, &v61, &v65, v43, v44, v49);
      LOBYTE(v16) = sub_BDD6D0((__int64 *)a1, (__int64)&v69);
      if ( *(_QWORD *)a1 )
      {
        sub_BDBD80(a1, (_BYTE *)a4);
        if ( v48 )
          sub_BDBD80(a1, v48);
        if ( v53 )
          sub_BDBD80(a1, v53);
        if ( v55 )
          sub_BD9900((__int64 *)a1, v55);
        if ( v42 )
          sub_BD9900((__int64 *)a1, (const char *)v42);
        LOBYTE(v16) = (unsigned __int8)sub_BD9900((__int64 *)a1, v22);
        v17 = (const char *)v51;
        if ( v51 )
          goto LABEL_18;
      }
      return v16;
    }
    v27 = sub_A17150(v50);
    LOBYTE(v16) = sub_BD9740(*((_BYTE **)v27 + 3));
    if ( !(_BYTE)v16 )
    {
      v45 = (const char *)*((_QWORD *)sub_A17150(v50) + 3);
      v69.m128i_i64[0] = (__int64)"invalid type ref";
      v71 = 259;
      LOBYTE(v16) = sub_BDD6D0((__int64 *)a1, (__int64)&v69);
      if ( *(_QWORD *)a1 )
      {
        LOBYTE(v16) = (_BYTE)v55;
        if ( v55 )
          LOBYTE(v16) = (unsigned __int8)sub_BD9900((__int64 *)a1, v55);
        if ( v45 )
          LOBYTE(v16) = (unsigned __int8)sub_BD9900((__int64 *)a1, v45);
      }
      return v16;
    }
    if ( !*(_BYTE *)(a1 + 825) )
      return v16;
    v28 = sub_B10CD0(a4 + 48);
    v29 = *(_BYTE *)(v28 - 16);
    if ( (v29 & 2) != 0 )
    {
      if ( *(_DWORD *)(v28 - 24) != 2 )
        goto LABEL_42;
      v16 = *(_QWORD *)(v28 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v28 - 16) >> 6) & 0xF) != 2 )
      {
LABEL_42:
        v16 = *(_QWORD *)(a4 + 32 * (1LL - (*(_DWORD *)(a4 + 4) & 0x7FFFFFF)));
        v30 = *(_QWORD *)(v16 + 24);
        if ( v30 )
        {
          v31 = *(unsigned __int16 *)(v30 + 20);
          if ( *(_WORD *)(v30 + 20) )
          {
            v32 = *(unsigned int *)(a1 + 1864);
            if ( (unsigned int)v31 > (unsigned int)v32 && v32 != v31 )
            {
              if ( v32 <= v31 )
              {
                v46 = v31 - v32;
                if ( v31 > *(unsigned int *)(a1 + 1868) )
                  sub_C8D5F0(a1 + 1856, a1 + 1872, *(unsigned __int16 *)(v30 + 20), 8);
                memset((void *)(*(_QWORD *)(a1 + 1856) + 8LL * *(unsigned int *)(a1 + 1864)), 0, 8 * v46);
                *(_DWORD *)(a1 + 1864) += v46;
              }
              else
              {
                *(_DWORD *)(a1 + 1864) = v31;
              }
            }
            v16 = *(_QWORD *)(a1 + 1856) + 8LL * (unsigned int)(v31 - 1);
            v33 = *(const char **)v16;
            *(_QWORD *)v16 = v30;
            if ( (const char *)v30 != v33 )
            {
              if ( v33 )
              {
                v69.m128i_i64[0] = (__int64)"conflicting debug info for argument";
                v71 = 259;
                LOBYTE(v16) = sub_BDD6D0((__int64 *)a1, (__int64)&v69);
                if ( *(_QWORD *)a1 )
                {
                  sub_BDBD80(a1, (_BYTE *)a4);
                  sub_BD9900((__int64 *)a1, v33);
                  LOBYTE(v16) = (unsigned __int8)sub_BD9900((__int64 *)a1, (const char *)v30);
                }
              }
            }
          }
        }
        else
        {
          v69.m128i_i64[0] = (__int64)"dbg intrinsic without variable";
          v71 = 259;
          LOBYTE(v16) = sub_BDD6D0((__int64 *)a1, (__int64)&v69);
        }
        return v16;
      }
      v16 = v28 - 16 - 8LL * ((v29 >> 2) & 0xF);
    }
    if ( *(_QWORD *)(v16 + 8) )
      return v16;
    goto LABEL_42;
  }
  HIBYTE(v71) = 1;
  v12 = "invalid llvm.dbg.assign intrinsic address";
LABEL_13:
  v13 = *(_QWORD *)a1;
  v69.m128i_i64[0] = (__int64)v12;
  LOBYTE(v71) = 3;
  if ( !v13 )
  {
    LOBYTE(v16) = *(_BYTE *)(a1 + 154);
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) |= v16;
    return v16;
  }
  sub_CA0E80(&v69, v13);
  v14 = *(_BYTE **)(v13 + 32);
  if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 24) )
  {
    sub_CB5D20(v13, 10);
  }
  else
  {
    *(_QWORD *)(v13 + 32) = v14 + 1;
    *v14 = 10;
  }
  v15 = *(_QWORD *)a1;
  LOBYTE(v16) = *(_BYTE *)(a1 + 154);
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 152) |= v16;
  if ( v15 )
  {
    sub_BDBD80(a1, (_BYTE *)a4);
    v17 = v10;
LABEL_18:
    LOBYTE(v16) = (unsigned __int8)sub_BD9900((__int64 *)a1, v17);
  }
  return v16;
}
