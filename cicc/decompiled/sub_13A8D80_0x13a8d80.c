// Function: sub_13A8D80
// Address: 0x13a8d80
//
__int64 __fastcall sub_13A8D80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // r10d
  unsigned int v15; // r14d
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // r10d
  unsigned __int64 v19; // rbx
  unsigned int v20; // r10d
  __int64 v21; // r12
  __int64 v22; // rdx
  unsigned int v23; // eax
  unsigned int v24; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // eax
  unsigned int v29; // eax
  __int64 *v30; // rdi
  unsigned int v31; // eax
  unsigned int v32; // eax
  __int64 *v33; // rdi
  int v34; // eax
  unsigned int v35; // eax
  unsigned int v36; // eax
  __int64 *v37; // rdi
  unsigned int v38; // eax
  __int64 *v39; // rdi
  int v40; // eax
  unsigned int v41; // eax
  __int64 *v42; // rdi
  int v43; // r8d
  char v44; // al
  unsigned int v45; // eax
  unsigned int v46; // eax
  __int64 *v47; // rdi
  int v48; // r8d
  char v49; // al
  __int64 v50; // rdx
  unsigned int v51; // eax
  unsigned int v52; // eax
  unsigned int v53; // eax
  unsigned int v54; // eax
  unsigned int v55; // eax
  unsigned int v56; // eax
  unsigned int v57; // eax
  unsigned int v58; // eax
  bool v59; // [rsp+1Ch] [rbp-194h]
  char v60; // [rsp+1Ch] [rbp-194h]
  unsigned int v61; // [rsp+50h] [rbp-160h]
  unsigned int v62; // [rsp+58h] [rbp-158h]
  unsigned int v63; // [rsp+58h] [rbp-158h]
  unsigned int v64; // [rsp+60h] [rbp-150h]
  char v65; // [rsp+60h] [rbp-150h]
  unsigned __int8 v66; // [rsp+68h] [rbp-148h]
  __int64 v67; // [rsp+70h] [rbp-140h] BYREF
  int v68; // [rsp+78h] [rbp-138h]
  __int64 v69; // [rsp+80h] [rbp-130h] BYREF
  int v70; // [rsp+88h] [rbp-128h]
  __int64 v71; // [rsp+90h] [rbp-120h] BYREF
  int v72; // [rsp+98h] [rbp-118h]
  __int64 v73; // [rsp+A0h] [rbp-110h] BYREF
  unsigned int v74; // [rsp+A8h] [rbp-108h]
  __int64 v75[2]; // [rsp+B0h] [rbp-100h] BYREF
  unsigned __int64 v76; // [rsp+C0h] [rbp-F0h] BYREF
  unsigned int v77; // [rsp+C8h] [rbp-E8h]
  unsigned __int64 v78; // [rsp+D0h] [rbp-E0h] BYREF
  unsigned int v79; // [rsp+D8h] [rbp-D8h]
  __int64 v80; // [rsp+E0h] [rbp-D0h] BYREF
  unsigned int v81; // [rsp+E8h] [rbp-C8h]
  __int64 v82[2]; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v83[2]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v84[2]; // [rsp+110h] [rbp-A0h] BYREF
  unsigned __int64 v85; // [rsp+120h] [rbp-90h] BYREF
  unsigned int v86; // [rsp+128h] [rbp-88h]
  unsigned __int64 v87; // [rsp+130h] [rbp-80h] BYREF
  unsigned int v88; // [rsp+138h] [rbp-78h]
  unsigned __int64 v89; // [rsp+140h] [rbp-70h] BYREF
  unsigned int v90; // [rsp+148h] [rbp-68h]
  __int64 v91[2]; // [rsp+150h] [rbp-60h] BYREF
  __int64 v92; // [rsp+160h] [rbp-50h] BYREF
  int v93; // [rsp+168h] [rbp-48h]
  __int64 v94; // [rsp+170h] [rbp-40h] BYREF
  int v95; // [rsp+178h] [rbp-38h]

  *(_BYTE *)(a8 + 43) = 0;
  v11 = sub_14806B0(*(_QWORD *)(a1 + 8), a5, a4, 0, 0);
  v12 = sub_1480620(*(_QWORD *)(a1 + 8), a3, 0);
  sub_13A62E0(a9, a2, v12, v11, a6);
  v66 = 0;
  if ( !*(_WORD *)(v11 + 24) && !*(_WORD *)(a2 + 24) && !*(_WORD *)(a3 + 24) )
  {
    v68 = 1;
    v13 = *(_QWORD *)(a2 + 32);
    v67 = 0;
    v70 = 1;
    v69 = 0;
    v72 = 1;
    v71 = 0;
    sub_13A38D0((__int64)&v73, v13 + 24);
    sub_13A38D0((__int64)v75, *(_QWORD *)(a3 + 32) + 24LL);
    v64 = v74;
    v66 = sub_13A3F30(
            v74,
            (__int64)&v73,
            (__int64)v75,
            *(_QWORD *)(v11 + 32) + 24LL,
            (__int64)&v67,
            (__int64)&v69,
            (__int64)&v71);
    if ( v66 )
    {
LABEL_22:
      sub_135E100(v75);
      sub_135E100(&v73);
      sub_135E100(&v71);
      sub_135E100(&v69);
      sub_135E100(&v67);
      return v66;
    }
    v14 = v64;
    v15 = v64 - 1;
    v77 = v64;
    v65 = (v64 - 1) & 0x3F;
    if ( v77 <= 0x40 )
    {
      v63 = v14;
      v19 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
      v76 = v19 & 1;
      v26 = sub_1456040(v11);
      v27 = sub_13A7B50(a1, a6, v26);
      v20 = v63;
      if ( v27 )
      {
        sub_13A36B0((__int64)&v76, (__int64 *)(*(_QWORD *)(v27 + 32) + 24LL));
        v20 = v63;
        v66 = 1;
      }
      v79 = v63;
      v21 = 1LL << v65;
      v22 = ~(1LL << v65);
    }
    else
    {
      v62 = v14;
      sub_16A4EF0(&v76, 1, 1);
      v16 = sub_1456040(v11);
      v17 = sub_13A7B50(a1, a6, v16);
      v18 = v62;
      if ( v17 )
      {
        sub_13A36B0((__int64)&v76, (__int64 *)(*(_QWORD *)(v17 + 32) + 24LL));
        v18 = v62;
        v66 = 1;
      }
      v79 = v62;
      v61 = v18;
      sub_16A4EF0(&v78, -1, 1);
      v19 = v78;
      v20 = v61;
      v21 = 1LL << v65;
      v22 = ~(1LL << v65);
      if ( v79 > 0x40 )
      {
        *(_QWORD *)(v78 + 8LL * (v15 >> 6)) &= v22;
        v81 = v61;
        goto LABEL_10;
      }
    }
    v81 = v20;
    v78 = v22 & v19;
    if ( v20 <= 0x40 )
    {
      v80 = 0;
      goto LABEL_29;
    }
LABEL_10:
    sub_16A4EF0(&v80, 0, 0);
    if ( v81 > 0x40 )
    {
      *(_QWORD *)(v80 + 8LL * (v15 >> 6)) |= v21;
      goto LABEL_12;
    }
LABEL_29:
    v80 |= v21;
LABEL_12:
    sub_16A9F90(v82, v75, &v67);
    if ( sub_13A39D0((__int64)v82, 0) )
    {
      sub_13A38D0((__int64)&v87, (__int64)&v69);
      if ( v88 > 0x40 )
        sub_16A8F40(&v87);
      else
        v87 = ~v87 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v88);
      sub_16A7400(&v87);
      v23 = v88;
      v88 = 0;
      v90 = v23;
      v89 = v87;
      sub_13A3A60((__int64)v91, (__int64)&v89, (__int64)v82);
      sub_13A38D0((__int64)&v92, (__int64)&v80);
      sub_13A37A0((__int64)&v94, (__int64)&v92, (__int64)v91);
      sub_13A3610(&v80, &v94);
      sub_135E100(&v94);
      sub_135E100(&v92);
      sub_135E100(v91);
      sub_135E100((__int64 *)&v89);
      sub_135E100((__int64 *)&v87);
      if ( !v66 )
        goto LABEL_16;
      sub_13A38D0((__int64)&v87, (__int64)&v76);
      sub_16A7590(&v87, &v69);
      v58 = v88;
      v88 = 0;
      v90 = v58;
      v89 = v87;
      sub_13A3C50((__int64)v91, (__int64)&v89, (__int64)v82);
      sub_13A38D0((__int64)&v92, (__int64)&v78);
      sub_13A3810((__int64)&v94, (__int64)&v92, (__int64)v91);
      v30 = (__int64 *)&v78;
    }
    else
    {
      sub_13A38D0((__int64)&v87, (__int64)&v69);
      if ( v88 > 0x40 )
        sub_16A8F40(&v87);
      else
        v87 = ~v87 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v88);
      sub_16A7400(&v87);
      v28 = v88;
      v88 = 0;
      v90 = v28;
      v89 = v87;
      sub_13A3C50((__int64)v91, (__int64)&v89, (__int64)v82);
      sub_13A38D0((__int64)&v92, (__int64)&v78);
      sub_13A3810((__int64)&v94, (__int64)&v92, (__int64)v91);
      sub_13A3610((__int64 *)&v78, &v94);
      sub_135E100(&v94);
      sub_135E100(&v92);
      sub_135E100(v91);
      sub_135E100((__int64 *)&v89);
      sub_135E100((__int64 *)&v87);
      if ( !v66 )
      {
LABEL_16:
        sub_16A9F90(&v94, &v73, &v67);
        sub_13A3610(v82, &v94);
        sub_135E100(&v94);
        if ( sub_13A39D0((__int64)v82, 0) )
        {
          sub_13A38D0((__int64)&v87, (__int64)&v71);
          if ( v88 > 0x40 )
            sub_16A8F40(&v87);
          else
            v87 = ~v87 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v88);
          sub_16A7400(&v87);
          v24 = v88;
          v88 = 0;
          v90 = v24;
          v89 = v87;
          sub_13A3A60((__int64)v91, (__int64)&v89, (__int64)v82);
          sub_13A38D0((__int64)&v92, (__int64)&v80);
          sub_13A37A0((__int64)&v94, (__int64)&v92, (__int64)v91);
          sub_13A3610(&v80, &v94);
          sub_135E100(&v94);
          sub_135E100(&v92);
          sub_135E100(v91);
          sub_135E100((__int64 *)&v89);
          sub_135E100((__int64 *)&v87);
          if ( !v66 )
            goto LABEL_20;
          sub_13A38D0((__int64)&v87, (__int64)&v76);
          sub_16A7590(&v87, &v71);
          v57 = v88;
          v88 = 0;
          v90 = v57;
          v89 = v87;
          sub_13A3C50((__int64)v91, (__int64)&v89, (__int64)v82);
          sub_13A38D0((__int64)&v92, (__int64)&v78);
          sub_13A3810((__int64)&v94, (__int64)&v92, (__int64)v91);
          v33 = (__int64 *)&v78;
        }
        else
        {
          sub_13A38D0((__int64)&v87, (__int64)&v71);
          if ( v88 > 0x40 )
            sub_16A8F40(&v87);
          else
            v87 = ~v87 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v88);
          sub_16A7400(&v87);
          v31 = v88;
          v88 = 0;
          v90 = v31;
          v89 = v87;
          sub_13A3C50((__int64)v91, (__int64)&v89, (__int64)v82);
          sub_13A38D0((__int64)&v92, (__int64)&v78);
          sub_13A3810((__int64)&v94, (__int64)&v92, (__int64)v91);
          sub_13A3610((__int64 *)&v78, &v94);
          sub_135E100(&v94);
          sub_135E100(&v92);
          sub_135E100(v91);
          sub_135E100((__int64 *)&v89);
          sub_135E100((__int64 *)&v87);
          if ( !v66 )
            goto LABEL_20;
          sub_13A38D0((__int64)&v87, (__int64)&v76);
          sub_16A7590(&v87, &v71);
          v32 = v88;
          v88 = 0;
          v90 = v32;
          v89 = v87;
          sub_13A3A60((__int64)v91, (__int64)&v89, (__int64)v82);
          sub_13A38D0((__int64)&v92, (__int64)&v80);
          sub_13A37A0((__int64)&v94, (__int64)&v92, (__int64)v91);
          v33 = &v80;
        }
        sub_13A3610(v33, &v94);
        sub_135E100(&v94);
        sub_135E100(&v92);
        sub_135E100(v91);
        sub_135E100((__int64 *)&v89);
        sub_135E100((__int64 *)&v87);
LABEL_20:
        v66 = 1;
        if ( (int)sub_16AEA10(&v80, &v78) <= 0 )
        {
          sub_13A38D0((__int64)v83, (__int64)&v78);
          sub_13A38D0((__int64)v84, (__int64)&v80);
          sub_13A38D0((__int64)&v92, (__int64)&v73);
          sub_16A7590(&v92, v75);
          v34 = v93;
          v93 = 0;
          v95 = v34;
          v94 = v92;
          sub_13A3610(v82, &v94);
          sub_135E100(&v94);
          sub_135E100(&v92);
          if ( sub_13A39D0((__int64)v82, 0) )
          {
            sub_13A38D0((__int64)&v85, (__int64)&v69);
            sub_16A7590(&v85, &v71);
            v51 = v86;
            v86 = 0;
            v88 = v51;
            v87 = v85;
            sub_16A7490(&v87, 1);
            v52 = v88;
            v88 = 0;
            v90 = v52;
            v89 = v87;
            sub_13A3A60((__int64)v91, (__int64)&v89, (__int64)v82);
            sub_13A38D0((__int64)&v92, (__int64)&v80);
            sub_13A37A0((__int64)&v94, (__int64)&v92, (__int64)v91);
            v37 = &v80;
          }
          else
          {
            sub_13A38D0((__int64)&v85, (__int64)&v69);
            sub_16A7590(&v85, &v71);
            v35 = v86;
            v86 = 0;
            v88 = v35;
            v87 = v85;
            sub_16A7490(&v87, 1);
            v36 = v88;
            v88 = 0;
            v90 = v36;
            v89 = v87;
            sub_13A3C50((__int64)v91, (__int64)&v89, (__int64)v82);
            sub_13A38D0((__int64)&v92, (__int64)&v78);
            sub_13A3810((__int64)&v94, (__int64)&v92, (__int64)v91);
            v37 = (__int64 *)&v78;
          }
          sub_13A3610(v37, &v94);
          sub_135E100(&v94);
          sub_135E100(&v92);
          sub_135E100(v91);
          sub_135E100((__int64 *)&v89);
          sub_135E100((__int64 *)&v87);
          sub_135E100((__int64 *)&v85);
          v59 = (int)sub_16AEA10(&v80, &v78) <= 0;
          sub_13A36B0((__int64)&v78, v83);
          sub_13A36B0((__int64)&v80, v84);
          if ( sub_13A39D0((__int64)v82, 0) )
          {
            sub_13A38D0((__int64)&v87, (__int64)&v69);
            sub_16A7590(&v87, &v71);
            v38 = v88;
            v88 = 0;
            v90 = v38;
            v89 = v87;
            sub_13A3A60((__int64)v91, (__int64)&v89, (__int64)v82);
            sub_13A38D0((__int64)&v92, (__int64)&v80);
            sub_13A37A0((__int64)&v94, (__int64)&v92, (__int64)v91);
            v39 = &v80;
          }
          else
          {
            sub_13A38D0((__int64)&v87, (__int64)&v69);
            sub_16A7590(&v87, &v71);
            v56 = v88;
            v88 = 0;
            v90 = v56;
            v89 = v87;
            sub_13A3C50((__int64)v91, (__int64)&v89, (__int64)v82);
            sub_13A38D0((__int64)&v92, (__int64)&v78);
            sub_13A3810((__int64)&v94, (__int64)&v92, (__int64)v91);
            v39 = (__int64 *)&v78;
          }
          sub_13A3610(v39, &v94);
          sub_135E100(&v94);
          sub_135E100(&v92);
          sub_135E100(v91);
          sub_135E100((__int64 *)&v89);
          sub_135E100((__int64 *)&v87);
          sub_13A38D0((__int64)&v92, (__int64)v75);
          sub_16A7590(&v92, &v73);
          v40 = v93;
          v93 = 0;
          v95 = v40;
          v94 = v92;
          sub_13A3610(v82, &v94);
          sub_135E100(&v94);
          sub_135E100(&v92);
          if ( sub_13A39D0((__int64)v82, 0) )
          {
            sub_13A38D0((__int64)&v87, (__int64)&v71);
            sub_16A7590(&v87, &v69);
            v41 = v88;
            v88 = 0;
            v90 = v41;
            v89 = v87;
            sub_13A3A60((__int64)v91, (__int64)&v89, (__int64)v82);
            sub_13A38D0((__int64)&v92, (__int64)&v80);
            sub_13A37A0((__int64)&v94, (__int64)&v92, (__int64)v91);
            v42 = &v80;
          }
          else
          {
            sub_13A38D0((__int64)&v87, (__int64)&v71);
            sub_16A7590(&v87, &v69);
            v55 = v88;
            v88 = 0;
            v90 = v55;
            v89 = v87;
            sub_13A3C50((__int64)v91, (__int64)&v89, (__int64)v82);
            sub_13A38D0((__int64)&v92, (__int64)&v78);
            sub_13A3810((__int64)&v94, (__int64)&v92, (__int64)v91);
            v42 = (__int64 *)&v78;
          }
          sub_13A3610(v42, &v94);
          sub_135E100(&v94);
          sub_135E100(&v92);
          sub_135E100(v91);
          sub_135E100((__int64 *)&v89);
          sub_135E100((__int64 *)&v87);
          v43 = sub_16AEA10(&v80, &v78);
          v44 = v59 | 2;
          if ( v43 > 0 )
            v44 = v59;
          v60 = v44;
          sub_13A36B0((__int64)&v78, v83);
          sub_13A36B0((__int64)&v80, v84);
          if ( sub_13A39D0((__int64)v82, 0) )
          {
            sub_13A38D0((__int64)&v85, (__int64)&v71);
            sub_16A7590(&v85, &v69);
            v45 = v86;
            v86 = 0;
            v88 = v45;
            v87 = v85;
            sub_16A7490(&v87, 1);
            v46 = v88;
            v88 = 0;
            v90 = v46;
            v89 = v87;
            sub_13A3A60((__int64)v91, (__int64)&v89, (__int64)v82);
            sub_13A38D0((__int64)&v92, (__int64)&v80);
            sub_13A37A0((__int64)&v94, (__int64)&v92, (__int64)v91);
            v47 = &v80;
          }
          else
          {
            sub_13A38D0((__int64)&v85, (__int64)&v71);
            sub_16A7590(&v85, &v69);
            v53 = v86;
            v86 = 0;
            v88 = v53;
            v87 = v85;
            sub_16A7490(&v87, 1);
            v54 = v88;
            v88 = 0;
            v90 = v54;
            v89 = v87;
            sub_13A3C50((__int64)v91, (__int64)&v89, (__int64)v82);
            sub_13A38D0((__int64)&v92, (__int64)&v78);
            sub_13A3810((__int64)&v94, (__int64)&v92, (__int64)v91);
            v47 = (__int64 *)&v78;
          }
          sub_13A3610(v47, &v94);
          sub_135E100(&v94);
          sub_135E100(&v92);
          sub_135E100(v91);
          sub_135E100((__int64 *)&v89);
          sub_135E100((__int64 *)&v87);
          sub_135E100((__int64 *)&v85);
          v48 = sub_16AEA10(&v80, &v78);
          v49 = v60 | 4;
          if ( v48 > 0 )
            v49 = v60;
          v50 = 16LL * (unsigned int)(a7 - 1);
          *(_BYTE *)(v50 + *(_QWORD *)(a8 + 48)) = v49 & *(_BYTE *)(v50 + *(_QWORD *)(a8 + 48)) & 7
                                                 | *(_BYTE *)(v50 + *(_QWORD *)(a8 + 48)) & 0xF8;
          v66 = (*(_BYTE *)(*(_QWORD *)(a8 + 48) + v50) & 7) == 0;
          sub_135E100(v84);
          sub_135E100(v83);
        }
        sub_135E100(v82);
        sub_135E100(&v80);
        sub_135E100((__int64 *)&v78);
        sub_135E100((__int64 *)&v76);
        goto LABEL_22;
      }
      sub_13A38D0((__int64)&v87, (__int64)&v76);
      sub_16A7590(&v87, &v69);
      v29 = v88;
      v88 = 0;
      v90 = v29;
      v89 = v87;
      sub_13A3A60((__int64)v91, (__int64)&v89, (__int64)v82);
      sub_13A38D0((__int64)&v92, (__int64)&v80);
      sub_13A37A0((__int64)&v94, (__int64)&v92, (__int64)v91);
      v30 = &v80;
    }
    sub_13A3610(v30, &v94);
    sub_135E100(&v94);
    sub_135E100(&v92);
    sub_135E100(v91);
    sub_135E100((__int64 *)&v89);
    sub_135E100((__int64 *)&v87);
    goto LABEL_16;
  }
  return v66;
}
