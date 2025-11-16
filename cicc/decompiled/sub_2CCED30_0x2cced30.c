// Function: sub_2CCED30
// Address: 0x2cced30
//
__int64 __fastcall sub_2CCED30(__int64 a1, __int64 a2)
{
  __int64 **v2; // rax
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  char v15; // bl
  __int64 v16; // rax
  _BYTE *v17; // rsi
  _BYTE *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // rsi
  _BYTE *v24; // rbx
  __int64 v25; // r12
  __int64 v26; // rbx
  unsigned __int16 v27; // bx
  char v28; // al
  __int64 v29; // rbx
  __int64 *v30; // rsi
  _BYTE *v31; // rax
  __int64 *v32; // rbx
  unsigned __int64 v33; // r12
  __int64 *v34; // r8
  __int64 v35; // r12
  __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // r14
  _QWORD *v39; // rax
  __int64 v40; // r15
  __int64 v41; // rsi
  __int64 v42; // rsi
  unsigned __int8 *v43; // rsi
  __int64 v44; // rbx
  __int64 v45; // rdx
  __int64 v46; // [rsp+8h] [rbp-148h]
  __int64 *v47; // [rsp+30h] [rbp-120h]
  __int64 v48; // [rsp+38h] [rbp-118h]
  __int64 *v49; // [rsp+40h] [rbp-110h]
  _QWORD *v50; // [rsp+48h] [rbp-108h]
  char v51; // [rsp+50h] [rbp-100h]
  __int64 v52; // [rsp+58h] [rbp-F8h]
  unsigned int v54; // [rsp+68h] [rbp-E8h]
  int v55; // [rsp+6Ch] [rbp-E4h]
  __int64 *v56; // [rsp+70h] [rbp-E0h]
  __int64 v57; // [rsp+70h] [rbp-E0h]
  __int64 *v58; // [rsp+70h] [rbp-E0h]
  __int64 v59; // [rsp+80h] [rbp-D0h]
  __int64 **v60; // [rsp+80h] [rbp-D0h]
  __int64 v61; // [rsp+88h] [rbp-C8h]
  unsigned __int64 v62; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v63; // [rsp+98h] [rbp-B8h]
  _BYTE *v64; // [rsp+A0h] [rbp-B0h]
  unsigned __int64 v65; // [rsp+B0h] [rbp-A0h] BYREF
  _BYTE *v66; // [rsp+B8h] [rbp-98h]
  _BYTE *v67; // [rsp+C0h] [rbp-90h]
  __int64 *v68; // [rsp+D0h] [rbp-80h] BYREF
  __int64 *v69; // [rsp+D8h] [rbp-78h]
  __int64 *v70; // [rsp+E0h] [rbp-70h]
  __int64 v71[4]; // [rsp+F0h] [rbp-60h] BYREF
  char v72; // [rsp+110h] [rbp-40h]
  char v73; // [rsp+111h] [rbp-3Fh]

  v2 = *(__int64 ***)(a2 + 40);
  v52 = (__int64)(v2 + 39);
  v47 = *v2;
  v3 = *(_QWORD *)(a2 + 80);
  if ( !v3 )
    BUG();
  v4 = *(_QWORD *)(v3 + 32);
  v5 = v4 - 24;
  if ( !v4 )
    v5 = 0;
  v46 = v5;
  v55 = 101;
  if ( (unsigned __int8)sub_CE9220(a2) )
  {
    v8 = a2;
    if ( (*(_BYTE *)(a2 + 2) & 1) == 0 )
    {
LABEL_6:
      v61 = *(_QWORD *)(a2 + 96);
      v9 = v61;
      goto LABEL_7;
    }
  }
  else
  {
    v55 = (_BYTE)qword_50139A8 == 0 ? 101 : 5;
    v8 = a2;
    if ( (*(_BYTE *)(a2 + 2) & 1) == 0 )
      goto LABEL_6;
  }
  v44 = v8;
  sub_B2C6D0(v8, a2, v6, v7);
  v61 = *(_QWORD *)(v44 + 96);
  if ( (*(_BYTE *)(v44 + 2) & 1) != 0 )
    sub_B2C6D0(v44, a2, v45, *(_QWORD *)(v44 + 96));
  v9 = *(_QWORD *)(v44 + 96);
LABEL_7:
  result = v9 + 40LL * *(_QWORD *)(a2 + 104);
  v48 = result;
  if ( result != v61 )
  {
LABEL_10:
    v11 = *(_QWORD *)(v61 + 8);
    v12 = sub_9208B0(v52, v11);
    v71[1] = v13;
    v71[0] = (unsigned __int64)(v12 + 7) >> 3;
    if ( sub_CA1930(v71) <= (unsigned __int64)(unsigned int)qword_5013708 )
      goto LABEL_9;
    v14 = sub_3936750();
    v15 = sub_314C600(a2, v14);
    sub_39367A0(v14);
    if ( v15 )
      goto LABEL_9;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v16 = sub_BCE3C0(v47, v55);
    v17 = v63;
    v71[0] = v16;
    if ( v63 == v64 )
    {
      sub_9183A0((__int64)&v62, v63, v71);
      v18 = v63;
    }
    else
    {
      if ( v63 )
      {
        *(_QWORD *)v63 = v16;
        v17 = v63;
      }
      v18 = v17 + 8;
      v63 = v18;
    }
    v19 = sub_B6E160(*(__int64 **)(a2 + 40), 0x1FE5u, v62, (__int64)&v18[-v62] >> 3);
    v65 = 0;
    v59 = v19;
    v66 = 0;
    v67 = 0;
    v20 = *(unsigned int *)(v61 + 32);
    v21 = sub_BCB2D0(v47);
    v22 = sub_ACD640(v21, v20, 0);
    v23 = v66;
    v71[0] = v22;
    if ( v66 == v67 )
    {
      sub_928380((__int64)&v65, v66, v71);
      v24 = v66;
    }
    else
    {
      if ( v66 )
      {
        *(_QWORD *)v66 = v22;
        v23 = v66;
      }
      v24 = v23 + 8;
      v66 = v23 + 8;
    }
    v73 = 1;
    v25 = 0;
    v72 = 3;
    v71[0] = (__int64)"ParamAddr";
    v56 = (__int64 *)v65;
    v26 = (__int64)&v24[-v65] >> 3;
    if ( v59 )
      v25 = *(_QWORD *)(v59 + 24);
    v50 = sub_BD2C40(88, (int)v26 + 1);
    if ( v50 )
    {
      v54 = (v26 + 1) & 0x7FFFFFF | v54 & 0xE0000000;
      sub_B44260((__int64)v50, **(_QWORD **)(v25 + 16), 56, v54, v46 + 24, 0);
      v50[9] = 0;
      sub_B4A290((__int64)v50, v25, v59, v56, v26, (__int64)v71, 0, 0);
    }
    v27 = sub_CE9380(a2, *(_DWORD *)(v61 + 32) + 1);
    v28 = sub_AE5020(v52, v11);
    v68 = 0;
    v69 = 0;
    if ( HIBYTE(v27) )
      v28 = v27;
    v70 = 0;
    v29 = *(_QWORD *)(v61 + 16);
    v51 = v28;
    if ( !v29 )
      goto LABEL_58;
    v30 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v31 = *(_BYTE **)(v29 + 24);
        if ( *v31 != 84 )
          break;
LABEL_31:
        v29 = *(_QWORD *)(v29 + 8);
        if ( !v29 )
          goto LABEL_35;
      }
      v71[0] = *(_QWORD *)(v29 + 24);
      if ( v70 != v30 )
      {
        if ( v30 )
        {
          *v30 = (__int64)v31;
          v30 = v69;
        }
        v69 = ++v30;
        goto LABEL_31;
      }
      sub_249A840((__int64)&v68, v30, v71);
      v29 = *(_QWORD *)(v29 + 8);
      v30 = v69;
      if ( !v29 )
      {
LABEL_35:
        v32 = v68;
        v33 = (unsigned __int64)v30;
        if ( v68 != v30 )
        {
          v60 = (__int64 **)v11;
          v49 = v30;
          while ( 1 )
          {
            v35 = *v32;
            v36 = sub_BCE760(v60, v55);
            v73 = 1;
            v57 = v36;
            v71[0] = (__int64)"bitCast";
            v72 = 3;
            v37 = sub_BD2C40(72, 1u);
            v38 = (__int64)v37;
            if ( v37 )
              sub_B51BF0((__int64)v37, (__int64)v50, v57, (__int64)v71, v35 + 24, 0);
            v71[0] = (__int64)"paramld";
            v73 = 1;
            v72 = 3;
            v39 = sub_BD2C40(80, 1u);
            v40 = (__int64)v39;
            if ( v39 )
              sub_B4D190((__int64)v39, (__int64)v60, v38, (__int64)v71, 0, v51, v35 + 24, 0);
            v41 = *(_QWORD *)(v35 + 48);
            v34 = (__int64 *)(v40 + 48);
            v71[0] = v41;
            if ( v41 )
              break;
            if ( v34 != v71 )
            {
              v42 = *(_QWORD *)(v40 + 48);
              if ( v42 )
                goto LABEL_48;
            }
LABEL_40:
            ++v32;
            sub_BD2ED0(v35, v61, v40);
            if ( v49 == v32 )
            {
              v33 = (unsigned __int64)v68;
              goto LABEL_56;
            }
          }
          sub_B96E90((__int64)v71, v41, 1);
          v34 = (__int64 *)(v40 + 48);
          if ( (__int64 *)(v40 + 48) == v71 )
          {
            if ( v71[0] )
              sub_B91220((__int64)v71, v71[0]);
            goto LABEL_40;
          }
          v42 = *(_QWORD *)(v40 + 48);
          if ( v42 )
          {
LABEL_48:
            v58 = v34;
            sub_B91220((__int64)v34, v42);
            v34 = v58;
          }
          v43 = (unsigned __int8 *)v71[0];
          *(_QWORD *)(v40 + 48) = v71[0];
          if ( v43 )
            sub_B976B0((__int64)v71, v43, (__int64)v34);
          goto LABEL_40;
        }
LABEL_56:
        if ( v33 )
          j_j___libc_free_0(v33);
LABEL_58:
        if ( v65 )
          j_j___libc_free_0(v65);
        if ( v62 )
          j_j___libc_free_0(v62);
LABEL_9:
        v61 += 40;
        result = v61;
        if ( v61 == v48 )
          return result;
        goto LABEL_10;
      }
    }
  }
  return result;
}
