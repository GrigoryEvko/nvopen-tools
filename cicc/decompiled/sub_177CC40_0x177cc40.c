// Function: sub_177CC40
// Address: 0x177cc40
//
void __fastcall sub_177CC40(
        __int64 a1,
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
  char v11; // al
  __int64 v12; // r14
  __int64 ***v13; // r13
  __int64 v14; // rsi
  __int64 *v15; // r14
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned __int8 *v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r15
  __int64 v21; // r14
  _QWORD *v22; // rax
  double v23; // xmm4_8
  double v24; // xmm5_8
  __int64 v25; // rsi
  __int64 v26; // r14
  int v27; // r9d
  __int64 v28; // rax
  _QWORD *v29; // r14
  __int64 v30; // rax
  unsigned __int64 v31; // r8
  int v32; // r15d
  unsigned __int8 *v33; // r10
  int v34; // r8d
  unsigned __int8 *v35; // rax
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // r14
  _QWORD *v39; // rax
  unsigned int v40; // r8d
  __int64 v41; // r10
  _QWORD *v42; // r13
  __int64 v43; // rax
  __int64 *v44; // rax
  __int64 *v45; // rax
  __int64 *v46; // r10
  int v47; // r8d
  __int64 *v48; // r11
  __int64 *v49; // rcx
  __int64 *v50; // rax
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rsi
  __int64 v54; // r15
  __int64 v55; // rsi
  unsigned __int8 *v56; // rsi
  __int64 v57; // rcx
  __int64 v58; // r14
  __int64 v59; // rax
  __int64 v60; // r15
  __int64 v61; // rsi
  __int64 *v62; // r14
  __int64 v63; // rsi
  __int64 v64; // rdx
  unsigned __int8 *v65; // rsi
  __int64 v66; // rcx
  __int64 *v67; // rax
  int v68; // [rsp+Ch] [rbp-D4h]
  int v69; // [rsp+10h] [rbp-D0h]
  __int64 *v70; // [rsp+18h] [rbp-C8h]
  __int64 *v71; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v72; // [rsp+20h] [rbp-C0h]
  __int64 v73; // [rsp+20h] [rbp-C0h]
  unsigned int v74; // [rsp+28h] [rbp-B8h]
  __int64 v75; // [rsp+28h] [rbp-B8h]
  __int64 v76; // [rsp+30h] [rbp-B0h]
  __int64 v77; // [rsp+30h] [rbp-B0h]
  __int64 v78; // [rsp+38h] [rbp-A8h]
  __int64 *v79; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v80[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v81; // [rsp+50h] [rbp-90h]
  unsigned __int8 *v82; // [rsp+60h] [rbp-80h] BYREF
  __int64 v83; // [rsp+68h] [rbp-78h]
  _WORD v84[56]; // [rsp+70h] [rbp-70h] BYREF

  v11 = *(_BYTE *)(a2 + 16);
  if ( v11 == 54 )
  {
    v12 = sub_1776680(a1, *(_QWORD *)(a2 - 24));
    v13 = (__int64 ***)sub_1648A60(64, 1u);
    if ( v13 )
      sub_15F9210((__int64)v13, *(_QWORD *)(*(_QWORD *)v12 + 24LL), v12, 0, 0, 0);
    sub_164B7C0((__int64)v13, a2);
    v14 = *(_QWORD *)(a2 + 48);
    v15 = *(__int64 **)(a1 + 104);
    v82 = (unsigned __int8 *)v14;
    if ( v14 )
    {
      sub_1623A60((__int64)&v82, v14, 2);
      v16 = (__int64)v13[6];
      v17 = (__int64)(v13 + 6);
      if ( !v16 )
        goto LABEL_7;
    }
    else
    {
      v16 = (__int64)v13[6];
      v17 = (__int64)(v13 + 6);
      if ( !v16 )
      {
LABEL_9:
        sub_157E9D0(*(_QWORD *)(a2 + 40) + 40LL, (__int64)v13);
        v19 = *(_QWORD *)(a2 + 24);
        v13[4] = (__int64 **)(a2 + 24);
        v19 &= 0xFFFFFFFFFFFFFFF8LL;
        v13[3] = (__int64 **)(v19 | (unsigned __int64)v13[3] & 7);
        *(_QWORD *)(v19 + 8) = v13 + 3;
        *(_QWORD *)(a2 + 24) = *(_QWORD *)(a2 + 24) & 7LL | (unsigned __int64)(v13 + 3);
        sub_170B990(*v15, (__int64)v13);
        v20 = *(_QWORD *)(a2 + 8);
        if ( v20 )
        {
          v21 = **(_QWORD **)(a1 + 104);
          do
          {
            v22 = sub_1648700(v20);
            sub_170B990(v21, (__int64)v22);
            v20 = *(_QWORD *)(v20 + 8);
          }
          while ( v20 );
          v25 = (__int64)v13;
          if ( v13 == (__int64 ***)a2 )
            v25 = sub_1599EF0(*v13);
          sub_164D160(a2, v25, a3, a4, a5, a6, v23, v24, a9, a10);
        }
        goto LABEL_15;
      }
    }
    v76 = v17;
    sub_161E7C0(v17, v16);
    v17 = v76;
LABEL_7:
    v18 = v82;
    v13[6] = (__int64 **)v82;
    if ( v18 )
      sub_1623210((__int64)&v82, v18, v17);
    goto LABEL_9;
  }
  if ( v11 != 56 )
  {
    v58 = sub_1776680(a1, *(_QWORD *)(a2 - 24));
    v59 = *(_QWORD *)v58;
    if ( *(_BYTE *)(*(_QWORD *)v58 + 8LL) == 16 )
      v59 = **(_QWORD **)(v59 + 16);
    v60 = sub_1646BA0(**(__int64 ***)(*(_QWORD *)a2 + 16LL), *(_DWORD *)(v59 + 8) >> 8);
    v84[0] = 257;
    v13 = (__int64 ***)sub_1648A60(56, 1u);
    if ( v13 )
      sub_15FD590((__int64)v13, v58, v60, (__int64)&v82, 0);
    v61 = *(_QWORD *)(a2 + 48);
    v62 = *(__int64 **)(a1 + 104);
    v82 = (unsigned __int8 *)v61;
    if ( v61 )
    {
      sub_1623A60((__int64)&v82, v61, 2);
      v63 = (__int64)v13[6];
      v64 = (__int64)(v13 + 6);
      if ( !v63 )
        goto LABEL_53;
    }
    else
    {
      v63 = (__int64)v13[6];
      v64 = (__int64)(v13 + 6);
      if ( !v63 )
      {
LABEL_55:
        sub_157E9D0(*(_QWORD *)(a2 + 40) + 40LL, (__int64)v13);
        v66 = *(_QWORD *)(a2 + 24);
        v13[4] = (__int64 **)(a2 + 24);
        v66 &= 0xFFFFFFFFFFFFFFF8LL;
        v13[3] = (__int64 **)(v66 | (unsigned __int64)v13[3] & 7);
        *(_QWORD *)(v66 + 8) = v13 + 3;
        *(_QWORD *)(a2 + 24) = *(_QWORD *)(a2 + 24) & 7LL | (unsigned __int64)(v13 + 3);
        sub_170B990(*v62, (__int64)v13);
        sub_164B7C0((__int64)v13, a2);
LABEL_15:
        v82 = (unsigned __int8 *)a2;
        *(_QWORD *)sub_177C990(a1 + 48, (unsigned __int64 *)&v82) = v13;
        return;
      }
    }
    v77 = v64;
    sub_161E7C0(v64, v63);
    v64 = v77;
LABEL_53:
    v65 = v82;
    v13[6] = (__int64 **)v82;
    if ( v65 )
      sub_1623210((__int64)&v82, v65, v64);
    goto LABEL_55;
  }
  v26 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v78 = sub_1776680(a1, *(_QWORD *)(a2 - 24 * v26));
  v82 = (unsigned __int8 *)v84;
  v83 = 0x800000000LL;
  v28 = 24 * (1 - v26);
  v29 = (_QWORD *)(a2 + v28);
  v30 = -v28;
  v31 = 0xAAAAAAAAAAAAAAABLL * (v30 >> 3);
  v32 = v31;
  if ( (unsigned __int64)v30 > 0xC0 )
  {
    sub_16CD150((__int64)&v82, v84, 0xAAAAAAAAAAAAAAABLL * (v30 >> 3), 8, v31, v27);
    v33 = v82;
    v34 = v83;
    v35 = &v82[8 * (unsigned int)v83];
  }
  else
  {
    v33 = (unsigned __int8 *)v84;
    v34 = 0;
    v35 = (unsigned __int8 *)v84;
  }
  if ( (_QWORD *)a2 != v29 )
  {
    do
    {
      if ( v35 )
        *(_QWORD *)v35 = *v29;
      v29 += 3;
      v35 += 8;
    }
    while ( (_QWORD *)a2 != v29 );
    v33 = v82;
    v34 = v83;
  }
  LODWORD(v83) = v32 + v34;
  v36 = (unsigned int)(v32 + v34);
  v81 = 257;
  v37 = *(_QWORD *)v78;
  v38 = **(_QWORD **)(*(_QWORD *)v78 + 16LL);
  if ( !v38 )
  {
    if ( *(_BYTE *)(v37 + 8) == 16 )
      BUG();
    v38 = *(_QWORD *)(v37 + 24);
  }
  v72 = v33;
  v74 = v83 + 1;
  v39 = sub_1648A60(72, (int)v83 + 1);
  v40 = v74;
  v41 = (__int64)v72;
  v42 = v39;
  if ( v39 )
  {
    v75 = (__int64)v39;
    v73 = (__int64)&v39[-3 * v40];
    v43 = *(_QWORD *)v78;
    if ( *(_BYTE *)(*(_QWORD *)v78 + 8LL) == 16 )
      v43 = **(_QWORD **)(v43 + 16);
    v68 = v40;
    v70 = (__int64 *)v41;
    v69 = *(_DWORD *)(v43 + 8) >> 8;
    v44 = (__int64 *)sub_15F9F50(v38, v41, v36);
    v45 = (__int64 *)sub_1646BA0(v44, v69);
    v46 = v70;
    v47 = v68;
    v48 = v45;
    if ( *(_BYTE *)(*(_QWORD *)v78 + 8LL) == 16 )
    {
      v67 = sub_16463B0(v45, *(_QWORD *)(*(_QWORD *)v78 + 32LL));
      v46 = v70;
      v47 = v68;
      v48 = v67;
    }
    else
    {
      v49 = &v70[v36];
      if ( v49 != v70 )
      {
        v50 = v70;
        while ( 1 )
        {
          v51 = *(_QWORD *)*v50;
          if ( *(_BYTE *)(v51 + 8) == 16 )
            break;
          if ( v49 == ++v50 )
            goto LABEL_37;
        }
        v52 = sub_16463B0(v48, *(_QWORD *)(v51 + 32));
        v47 = v68;
        v46 = v70;
        v48 = v52;
      }
    }
LABEL_37:
    v71 = v46;
    sub_15F1EA0((__int64)v42, (__int64)v48, 32, v73, v47, 0);
    v42[7] = v38;
    v42[8] = sub_15F9F50(v38, (__int64)v71, v36);
    sub_15F9CE0((__int64)v42, v78, v71, v36, (__int64)v80);
  }
  else
  {
    v75 = 0;
  }
  v53 = *(_QWORD *)(a2 + 48);
  v79 = *(__int64 **)(a1 + 104);
  v80[0] = v53;
  if ( v53 )
  {
    v54 = (__int64)(v42 + 6);
    sub_1623A60((__int64)v80, v53, 2);
    v55 = v42[6];
    if ( !v55 )
      goto LABEL_41;
    goto LABEL_40;
  }
  v55 = v42[6];
  v54 = (__int64)(v42 + 6);
  if ( v55 )
  {
LABEL_40:
    sub_161E7C0(v54, v55);
LABEL_41:
    v56 = (unsigned __int8 *)v80[0];
    v42[6] = v80[0];
    if ( v56 )
      sub_1623210((__int64)v80, v56, v54);
  }
  sub_157E9D0(*(_QWORD *)(a2 + 40) + 40LL, (__int64)v42);
  v57 = *(_QWORD *)(a2 + 24);
  v42[4] = a2 + 24;
  v57 &= 0xFFFFFFFFFFFFFFF8LL;
  v42[3] = v57 | v42[3] & 7LL;
  *(_QWORD *)(v57 + 8) = v42 + 3;
  *(_QWORD *)(a2 + 24) = *(_QWORD *)(a2 + 24) & 7LL | (unsigned __int64)(v42 + 3);
  sub_170B990(*v79, (__int64)v42);
  sub_164B7C0(v75, a2);
  v80[0] = a2;
  *(_QWORD *)sub_177C990(a1 + 48, v80) = v42;
  if ( v82 != (unsigned __int8 *)v84 )
    _libc_free((unsigned __int64)v82);
}
