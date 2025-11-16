// Function: sub_17C5C60
// Address: 0x17c5c60
//
unsigned __int64 __fastcall sub_17C5C60(
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
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rcx
  unsigned int v16; // esi
  __int64 *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r12
  __int64 v20; // rcx
  __int64 v21; // rdx
  _QWORD *v22; // rdi
  __int64 v23; // rdx
  _QWORD *v24; // rax
  unsigned __int8 *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // rcx
  unsigned __int8 *v30; // rax
  __int64 **v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rsi
  _QWORD **v35; // rdi
  __int64 v36; // rax
  __int64 v37; // r12
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 v40; // rax
  char v41; // bl
  unsigned __int64 result; // rax
  int v43; // eax
  __int64 *v44; // rax
  __int64 **v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // r12
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rsi
  _QWORD **v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 v63; // rdx
  unsigned __int8 *v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rdx
  unsigned __int8 *v70; // rsi
  int v71; // r11d
  __int64 *v72; // [rsp+10h] [rbp-110h]
  __int64 *v73; // [rsp+10h] [rbp-110h]
  __int64 v74; // [rsp+18h] [rbp-108h]
  unsigned __int8 *v75; // [rsp+28h] [rbp-F8h] BYREF
  __int64 v76[2]; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v77; // [rsp+40h] [rbp-E0h]
  __int64 v78[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int16 v79; // [rsp+60h] [rbp-C0h]
  unsigned __int8 *v80[2]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v81; // [rsp+80h] [rbp-A0h]
  __int64 v82; // [rsp+88h] [rbp-98h]
  __int64 v83; // [rsp+90h] [rbp-90h]
  __int64 v84; // [rsp+98h] [rbp-88h]
  unsigned __int8 *v85; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v86; // [rsp+A8h] [rbp-78h]
  __int64 *v87; // [rsp+B0h] [rbp-70h]
  _QWORD *v88; // [rsp+B8h] [rbp-68h]
  __int64 v89; // [rsp+C0h] [rbp-60h]
  int v90; // [rsp+C8h] [rbp-58h]
  __int64 v91; // [rsp+D0h] [rbp-50h]
  __int64 v92; // [rsp+D8h] [rbp-48h]

  v12 = sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v13 = *(unsigned int *)(a1 + 136);
  v14 = *(_QWORD *)(a1 + 120);
  if ( (_DWORD)v13 )
  {
    v15 = v12;
    v16 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v17 = (__int64 *)(v14 + 32LL * v16);
    v18 = *v17;
    if ( v15 == *v17 )
      goto LABEL_3;
    v43 = 1;
    while ( v18 != -8 )
    {
      v71 = v43 + 1;
      v16 = (v13 - 1) & (v43 + v16);
      v17 = (__int64 *)(v14 + 32LL * v16);
      v18 = *v17;
      if ( v15 == *v17 )
        goto LABEL_3;
      v43 = v71;
    }
  }
  v17 = (__int64 *)(v14 + 32 * v13);
LABEL_3:
  v19 = v17[3];
  v20 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v21 = *(_QWORD *)(a2 + 24 * (3 - v20));
  v22 = *(_QWORD **)(v21 + 24);
  if ( *(_DWORD *)(v21 + 32) > 0x40u )
    v22 = (_QWORD *)*v22;
  v23 = *(_QWORD *)(a2 + 24 * (4 - v20));
  if ( *(_DWORD *)(v23 + 32) <= 0x40u )
    v74 = *(_QWORD *)(v23 + 24);
  else
    v74 = **(_QWORD **)(v23 + 24);
  if ( v22 )
  {
    LODWORD(v74) = *((_DWORD *)v17 + 2) + v74;
    if ( v22 != (_QWORD *)1 )
      LODWORD(v74) = *((_DWORD *)v17 + 3) + v74;
  }
  v24 = (_QWORD *)sub_16498A0(a2);
  v25 = *(unsigned __int8 **)(a2 + 48);
  v85 = 0;
  v88 = v24;
  v26 = *(_QWORD *)(a2 + 40);
  v89 = 0;
  v86 = v26;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v87 = (__int64 *)(a2 + 24);
  v80[0] = v25;
  if ( v25 )
  {
    sub_1623A60((__int64)v80, (__int64)v25, 2);
    if ( v85 )
      sub_161E7C0((__int64)&v85, (__int64)v85);
    v85 = v80[0];
    if ( v80[0] )
      sub_1623210((__int64)v80, v80[0], (__int64)&v85);
  }
  v27 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v28 = *(_QWORD *)(a2 + 24 * (3 - v27));
  v29 = *(_QWORD **)(v28 + 24);
  if ( *(_DWORD *)(v28 + 32) > 0x40u )
    v29 = (_QWORD *)*v29;
  v30 = *(unsigned __int8 **)(a2 + 24 * (2 - v27));
  if ( v29 == (_QWORD *)1 )
  {
    v80[0] = v30;
    LOWORD(v77) = 257;
    v45 = (__int64 **)sub_16471D0(v88, 0);
    if ( v45 != *(__int64 ***)v19 )
    {
      if ( *(_BYTE *)(v19 + 16) > 0x10u )
      {
        v79 = 257;
        v65 = sub_15FDBD0(47, v19, (__int64)v45, (__int64)v78, 0);
        v19 = v65;
        if ( v86 )
        {
          v73 = v87;
          sub_157E9D0(v86 + 40, v65);
          v66 = *v73;
          v67 = *(_QWORD *)(v19 + 24) & 7LL;
          *(_QWORD *)(v19 + 32) = v73;
          v66 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v19 + 24) = v66 | v67;
          *(_QWORD *)(v66 + 8) = v19 + 24;
          *v73 = *v73 & 7 | (v19 + 24);
        }
        sub_164B780(v19, v76);
        if ( v85 )
        {
          v75 = v85;
          sub_1623A60((__int64)&v75, (__int64)v85, 2);
          v68 = *(_QWORD *)(v19 + 48);
          v69 = v19 + 48;
          if ( v68 )
          {
            sub_161E7C0(v19 + 48, v68);
            v69 = v19 + 48;
          }
          v70 = v75;
          *(_QWORD *)(v19 + 48) = v75;
          if ( v70 )
            sub_1623210((__int64)&v75, v70, v69);
        }
      }
      else
      {
        v19 = sub_15A46C0(47, (__int64 ***)v19, v45, 0);
      }
    }
    v80[1] = (unsigned __int8 *)v19;
    v46 = sub_1643350(v88);
    v47 = sub_159C470(v46, (unsigned int)v74, 0);
    v48 = *(_QWORD *)(a1 + 232);
    v81 = v47;
    v49 = sub_1643360(v88);
    v50 = sub_159C470(v49, v48, 0);
    v51 = *(_QWORD *)(a1 + 240);
    v82 = v50;
    v52 = sub_1643360(v88);
    v83 = sub_159C470(v52, v51, 0);
    v53 = LODWORD(qword_4FA3C60[20]);
    if ( !(_DWORD)v53 )
      v53 = 0x8000000000000000LL;
    v54 = sub_1643360(v88);
    v55 = sub_159C470(v54, v53, 0);
    v56 = *(_QWORD *)(a1 + 104);
    v57 = *(_QWORD ***)(a1 + 40);
    v84 = v55;
    v79 = 257;
    v58 = sub_17C4C30(v57, v56, 1);
    v37 = sub_1285290((__int64 *)&v85, *(_QWORD *)(*(_QWORD *)v58 + 24LL), v58, (int)v80, 6, (__int64)v78, 0);
  }
  else
  {
    v76[0] = (__int64)v30;
    v79 = 257;
    v31 = (__int64 **)sub_16471D0(v88, 0);
    if ( v31 != *(__int64 ***)v19 )
    {
      if ( *(_BYTE *)(v19 + 16) > 0x10u )
      {
        LOWORD(v81) = 257;
        v59 = sub_15FDBD0(47, v19, (__int64)v31, (__int64)v80, 0);
        v19 = v59;
        if ( v86 )
        {
          v72 = v87;
          sub_157E9D0(v86 + 40, v59);
          v60 = *v72;
          v61 = *(_QWORD *)(v19 + 24) & 7LL;
          *(_QWORD *)(v19 + 32) = v72;
          v60 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v19 + 24) = v60 | v61;
          *(_QWORD *)(v60 + 8) = v19 + 24;
          *v72 = *v72 & 7 | (v19 + 24);
        }
        sub_164B780(v19, v78);
        if ( v85 )
        {
          v75 = v85;
          sub_1623A60((__int64)&v75, (__int64)v85, 2);
          v62 = *(_QWORD *)(v19 + 48);
          v63 = v19 + 48;
          if ( v62 )
          {
            sub_161E7C0(v19 + 48, v62);
            v63 = v19 + 48;
          }
          v64 = v75;
          *(_QWORD *)(v19 + 48) = v75;
          if ( v64 )
            sub_1623210((__int64)&v75, v64, v63);
        }
      }
      else
      {
        v19 = sub_15A46C0(47, (__int64 ***)v19, v31, 0);
      }
    }
    v76[1] = v19;
    v32 = sub_1643350(v88);
    v33 = sub_159C470(v32, (unsigned int)v74, 0);
    v34 = *(_QWORD *)(a1 + 104);
    v35 = *(_QWORD ***)(a1 + 40);
    v77 = v33;
    LOWORD(v81) = 257;
    v36 = sub_17C4C30(v35, v34, 0);
    v37 = sub_1285290((__int64 *)&v85, *(_QWORD *)(*(_QWORD *)v36 + 24LL), v36, (int)v76, 3, (__int64)v80, 0);
  }
  v40 = **(_QWORD **)(a1 + 104);
  if ( *(_BYTE *)(v40 + 144) )
  {
    v41 = 58;
  }
  else
  {
    v41 = 40;
    if ( !*(_BYTE *)(v40 + 146) )
      goto LABEL_24;
  }
  v80[0] = *(unsigned __int8 **)(v37 + 56);
  v44 = (__int64 *)sub_16498A0(v37);
  *(_QWORD *)(v37 + 56) = sub_1563AB0((__int64 *)v80, v44, 3, v41);
LABEL_24:
  sub_164D160(a2, v37, a3, a4, a5, a6, v38, v39, a9, a10);
  result = sub_15F20C0((_QWORD *)a2);
  if ( v85 )
    return sub_161E7C0((__int64)&v85, (__int64)v85);
  return result;
}
