// Function: sub_1860BE0
// Address: 0x1860be0
//
void __fastcall sub_1860BE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  char v14; // al
  __int64 ***v15; // r14
  __int16 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rbx
  __int64 v20; // rdx
  _QWORD *v21; // r13
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 **v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // rsi
  __int64 v30; // rax
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // r12
  __int64 v36; // rax
  unsigned __int64 v37; // r14
  _QWORD *v38; // rax
  __int64 v39; // rcx
  int v40; // edx
  _QWORD *v41; // rax
  const char *v42; // rax
  __int64 v43; // r13
  __int64 v44; // r14
  __int64 *v45; // r12
  __int64 v46; // rdx
  _QWORD *v47; // rax
  double v48; // xmm4_8
  double v49; // xmm5_8
  __int64 v50; // rbx
  int v51; // r8d
  __int64 v52; // rax
  __int64 *v53; // rax
  __int64 *v54; // rax
  int v55; // r8d
  __int64 *v56; // r10
  __int64 *v57; // rcx
  __int64 *v58; // rax
  __int64 v59; // rdx
  __int64 *v60; // rax
  unsigned int v61; // esi
  __int64 v62; // r8
  unsigned int v63; // ebx
  unsigned int v64; // edi
  __int64 *v65; // rax
  __int64 v66; // rcx
  int v67; // r11d
  __int64 *v68; // rdx
  int v69; // eax
  int v70; // ecx
  __int64 v71; // rbx
  __int64 v72; // rdi
  _QWORD *v73; // rax
  __int64 v74; // rax
  __int64 *v75; // rax
  int v76; // eax
  int v77; // r9d
  __int64 v78; // r11
  unsigned int v79; // eax
  __int64 v80; // r8
  int v81; // edi
  __int64 *v82; // rsi
  int v83; // eax
  int v84; // edi
  __int64 v85; // r8
  __int64 *v86; // r9
  unsigned int v87; // ebx
  int v88; // eax
  __int64 v89; // rsi
  int v90; // [rsp+0h] [rbp-D0h]
  int v91; // [rsp+4h] [rbp-CCh]
  unsigned int v92; // [rsp+8h] [rbp-C8h]
  __int64 v93; // [rsp+8h] [rbp-C8h]
  __int64 *v94; // [rsp+18h] [rbp-B8h]
  __int64 v95; // [rsp+18h] [rbp-B8h]
  _QWORD v96[2]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v97[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v98; // [rsp+40h] [rbp-90h]
  _QWORD *v99; // [rsp+50h] [rbp-80h] BYREF
  __int64 v100; // [rsp+58h] [rbp-78h]
  _QWORD v101[14]; // [rsp+60h] [rbp-70h] BYREF

  v14 = *(_BYTE *)(a1 + 16);
  if ( v14 == 75 )
  {
    v15 = (__int64 ***)sub_1860630(*(_QWORD *)(a1 - 48), 0, a2, a3);
    v16 = *(_WORD *)(a1 + 18) & 0x7FFF;
    v19 = sub_15A06D0(*v15, 0, v17, v18);
    v97[0] = sub_1649960(a1);
    LOWORD(v101[0]) = 261;
    v97[1] = v20;
    v99 = v97;
    v21 = sub_1648A60(56, 2u);
    if ( v21 )
    {
      v24 = *v15;
      if ( *((_BYTE *)*v15 + 8) == 16 )
      {
        v94 = v24[4];
        v25 = (__int64 *)sub_1643320(*v24);
        v26 = (__int64)sub_16463B0(v25, (unsigned int)v94);
      }
      else
      {
        v26 = sub_1643320(*v24);
      }
      sub_15FEC10((__int64)v21, v26, 51, v16, (__int64)v15, v19, (__int64)&v99, a1);
    }
    sub_164D160(a1, (__int64)v21, a4, a5, a6, a7, v22, v23, a10, a11);
    sub_15F20C0((_QWORD *)a1);
    return;
  }
  if ( v14 != 56 )
  {
    v61 = *(_DWORD *)(a2 + 24);
    if ( v61 )
    {
      v62 = *(_QWORD *)(a2 + 8);
      v63 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
      v64 = (v61 - 1) & v63;
      v65 = (__int64 *)(v62 + 32LL * v64);
      v66 = *v65;
      if ( a1 == *v65 )
        return;
      v67 = 1;
      v68 = 0;
      while ( v66 != -8 )
      {
        if ( v68 || v66 != -16 )
          v65 = v68;
        v64 = (v61 - 1) & (v67 + v64);
        v66 = *(_QWORD *)(v62 + 32LL * v64);
        if ( a1 == v66 )
          return;
        ++v67;
        v68 = v65;
        v65 = (__int64 *)(v62 + 32LL * v64);
      }
      if ( !v68 )
        v68 = v65;
      v69 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      v70 = v69 + 1;
      if ( 4 * (v69 + 1) < 3 * v61 )
      {
        if ( v61 - *(_DWORD *)(a2 + 20) - v70 > v61 >> 3 )
        {
LABEL_41:
          *(_DWORD *)(a2 + 16) = v70;
          if ( *v68 != -8 )
            --*(_DWORD *)(a2 + 20);
          *v68 = a1;
          v68[1] = 0;
          v68[2] = 0;
          v68[3] = 0;
          v71 = *(_QWORD *)(a1 + 8);
          while ( v71 )
          {
            v72 = v71;
            v71 = *(_QWORD *)(v71 + 8);
            v73 = sub_1648700(v72);
            sub_1860BE0(v73, a2, a3);
          }
          return;
        }
        sub_1860410(a2, v61);
        v83 = *(_DWORD *)(a2 + 24);
        if ( v83 )
        {
          v84 = v83 - 1;
          v85 = *(_QWORD *)(a2 + 8);
          v86 = 0;
          v87 = (v83 - 1) & v63;
          v70 = *(_DWORD *)(a2 + 16) + 1;
          v88 = 1;
          v68 = (__int64 *)(v85 + 32LL * v87);
          v89 = *v68;
          if ( a1 != *v68 )
          {
            while ( v89 != -8 )
            {
              if ( !v86 && v89 == -16 )
                v86 = v68;
              v87 = v84 & (v87 + v88);
              v68 = (__int64 *)(v85 + 32LL * v87);
              v89 = *v68;
              if ( a1 == *v68 )
                goto LABEL_41;
              ++v88;
            }
            if ( v86 )
              v68 = v86;
          }
          goto LABEL_41;
        }
LABEL_79:
        ++*(_DWORD *)(a2 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a2;
    }
    sub_1860410(a2, 2 * v61);
    v76 = *(_DWORD *)(a2 + 24);
    if ( v76 )
    {
      v77 = v76 - 1;
      v78 = *(_QWORD *)(a2 + 8);
      v79 = (v76 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v70 = *(_DWORD *)(a2 + 16) + 1;
      v68 = (__int64 *)(v78 + 32LL * v79);
      v80 = *v68;
      if ( a1 != *v68 )
      {
        v81 = 1;
        v82 = 0;
        while ( v80 != -8 )
        {
          if ( !v82 && v80 == -16 )
            v82 = v68;
          v79 = v77 & (v81 + v79);
          v68 = (__int64 *)(v78 + 32LL * v79);
          v80 = *v68;
          if ( a1 == *v68 )
            goto LABEL_41;
          ++v81;
        }
        if ( v82 )
          v68 = v82;
      }
      goto LABEL_41;
    }
    goto LABEL_79;
  }
  v27 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v28 = *(_QWORD *)(a1 + 24 * (2 - v27));
  v29 = *(_QWORD **)(v28 + 24);
  if ( *(_DWORD *)(v28 + 32) > 0x40u )
    v29 = (_QWORD *)*v29;
  v30 = sub_1860630(*(_QWORD *)(a1 - 24 * v27), (unsigned __int32)v29, a2, a3);
  v100 = 0x800000001LL;
  v95 = v30;
  v99 = v101;
  v33 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v101[0] = *(_QWORD *)(a1 + 24 * (1 - v33));
  v34 = 72 - 24 * v33;
  v35 = (_QWORD *)(a1 + v34);
  v36 = -v34;
  v37 = 0xAAAAAAAAAAAAAAABLL * (v36 >> 3);
  if ( (unsigned __int64)v36 > 0xA8 )
  {
    sub_16CD150((__int64)&v99, v101, v37 + 1, 8, v31, v32);
    v40 = v100;
    v38 = v99;
    v39 = (unsigned int)v100;
  }
  else
  {
    v38 = v101;
    v39 = 1;
    v40 = 1;
  }
  v41 = &v38[v39];
  if ( v35 != (_QWORD *)a1 )
  {
    do
    {
      if ( v41 )
        *v41 = *v35;
      v35 += 3;
      ++v41;
    }
    while ( (_QWORD *)a1 != v35 );
    v40 = v100;
  }
  LODWORD(v100) = v40 + v37;
  v42 = sub_1649960(a1);
  v43 = (unsigned int)v100;
  v44 = *(_QWORD *)(a1 + 64);
  v96[0] = v42;
  v45 = v99;
  v98 = 261;
  v96[1] = v46;
  v97[0] = v96;
  if ( !v44 )
  {
    v74 = *(_QWORD *)v95;
    if ( *(_BYTE *)(*(_QWORD *)v95 + 8LL) == 16 )
      v74 = **(_QWORD **)(v74 + 16);
    v44 = *(_QWORD *)(v74 + 24);
  }
  v92 = v100 + 1;
  v47 = sub_1648A60(72, (int)v100 + 1);
  v50 = (__int64)v47;
  if ( v47 )
  {
    v51 = v92;
    v93 = (__int64)&v47[-3 * v92];
    v52 = *(_QWORD *)v95;
    if ( *(_BYTE *)(*(_QWORD *)v95 + 8LL) == 16 )
      v52 = **(_QWORD **)(v52 + 16);
    v90 = v51;
    v91 = *(_DWORD *)(v52 + 8) >> 8;
    v53 = (__int64 *)sub_15F9F50(v44, (__int64)v45, v43);
    v54 = (__int64 *)sub_1646BA0(v53, v91);
    v55 = v90;
    v56 = v54;
    if ( *(_BYTE *)(*(_QWORD *)v95 + 8LL) == 16 )
    {
      v75 = sub_16463B0(v54, *(_QWORD *)(*(_QWORD *)v95 + 32LL));
      v55 = v90;
      v56 = v75;
    }
    else
    {
      v57 = &v45[v43];
      if ( v45 != v57 )
      {
        v58 = v45;
        while ( 1 )
        {
          v59 = *(_QWORD *)*v58;
          if ( *(_BYTE *)(v59 + 8) == 16 )
            break;
          if ( v57 == ++v58 )
            goto LABEL_28;
        }
        v60 = sub_16463B0(v56, *(_QWORD *)(v59 + 32));
        v55 = v90;
        v56 = v60;
      }
    }
LABEL_28:
    sub_15F1EA0(v50, (__int64)v56, 32, v93, v55, a1);
    *(_QWORD *)(v50 + 56) = v44;
    *(_QWORD *)(v50 + 64) = sub_15F9F50(v44, (__int64)v45, v43);
    sub_15F9CE0(v50, v95, v45, v43, (__int64)v97);
  }
  sub_164D160(a1, v50, a4, a5, a6, a7, v48, v49, a10, a11);
  sub_15F20C0((_QWORD *)a1);
  if ( v99 != v101 )
    _libc_free((unsigned __int64)v99);
}
