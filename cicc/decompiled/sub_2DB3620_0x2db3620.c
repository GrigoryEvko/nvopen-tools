// Function: sub_2DB3620
// Address: 0x2db3620
//
void __fastcall sub_2DB3620(__int64 *a1, _BYTE *a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // r12
  _BYTE *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int64 *v11; // r13
  __int64 v12; // rax
  unsigned __int64 *v13; // r15
  unsigned __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rbx
  unsigned __int64 *v17; // r12
  __int64 v18; // rax
  __int64 *v19; // r13
  unsigned __int64 *v20; // r15
  unsigned __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rsi
  int *v25; // r15
  int *v26; // r12
  int *v27; // r13
  __int64 v28; // rbx
  __int64 v29; // rdi
  unsigned int v30; // r15d
  __int64 v31; // rax
  __int64 v32; // rcx
  unsigned int v33; // r8d
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // rdi
  unsigned __int8 *v43; // rsi
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rbx
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 v53; // rax
  unsigned __int8 *v54; // rsi
  _QWORD *v55; // rbx
  int *v56; // rax
  __int64 v57; // rdx
  int *v58; // r13
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // r15
  __int64 v62; // rax
  unsigned __int64 *v63; // r8
  unsigned __int64 *v64; // r8
  unsigned __int64 v65; // rax
  unsigned __int64 v66; // rdx
  int v67; // eax
  __int64 v68; // rdi
  __int64 v69; // rdx
  __int64 v70; // r12
  __int64 v71; // r13
  __int64 *v72; // rbx
  __int64 v73; // rcx
  unsigned __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rbx
  __int64 v79; // rax
  __int64 v80; // rax
  int v81; // [rsp+Ch] [rbp-D4h]
  __int64 v82; // [rsp+10h] [rbp-D0h]
  __int64 v83; // [rsp+18h] [rbp-C8h]
  __int64 v84; // [rsp+18h] [rbp-C8h]
  unsigned __int64 *v85; // [rsp+18h] [rbp-C8h]
  unsigned __int64 *v86; // [rsp+18h] [rbp-C8h]
  __int64 *v87; // [rsp+20h] [rbp-C0h]
  int *v88; // [rsp+20h] [rbp-C0h]
  __int64 v89; // [rsp+28h] [rbp-B8h]
  __int64 v90; // [rsp+30h] [rbp-B0h]
  unsigned __int64 *v91; // [rsp+40h] [rbp-A0h]
  unsigned int v92; // [rsp+40h] [rbp-A0h]
  unsigned __int8 *v93; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int8 *v94; // [rsp+58h] [rbp-88h] BYREF
  unsigned __int8 *v95; // [rsp+60h] [rbp-80h] BYREF
  __int64 v96; // [rsp+68h] [rbp-78h]
  __int64 v97; // [rsp+70h] [rbp-70h]
  unsigned __int64 v98; // [rsp+80h] [rbp-60h] BYREF
  __int64 v99; // [rsp+88h] [rbp-58h]
  __int64 v100; // [rsp+90h] [rbp-50h] BYREF
  __int64 v101; // [rsp+98h] [rbp-48h]
  __int64 v102; // [rsp+A0h] [rbp-40h]

  v7 = a3;
  v8 = (_BYTE *)a1[5];
  v9 = a1[4];
  v89 = (__int64)a2;
  if ( v8 != (_BYTE *)v9 )
  {
    if ( (_BYTE)a3 )
    {
      a2 = v8;
      sub_2DB32C0(a1, v8, 0, a4, a5, a6);
      v8 = (_BYTE *)a1[5];
    }
    v10 = a1[3];
    v11 = (unsigned __int64 *)sub_2E313E0(v8, a2, a3, a4, a5);
    v12 = a1[5];
    a5 = *(_QWORD *)(v12 + 56);
    if ( (unsigned __int64 *)a5 != v11 )
    {
      v13 = (unsigned __int64 *)a1[92];
      if ( v11 != v13 )
      {
        a2 = (_BYTE *)(v12 + 40);
        v87 = *(__int64 **)(v12 + 56);
        sub_2E310C0(v10 + 40, v12 + 40, v87, v11);
        a5 = (__int64)v87;
        if ( v87 != (__int64 *)v11 )
        {
          a4 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*v87 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v11;
          *v11 = *v11 & 7 | *v87 & 0xFFFFFFFFFFFFFFF8LL;
          v14 = *v13;
          *(_QWORD *)(a4 + 8) = v13;
          v14 &= 0xFFFFFFFFFFFFFFF8LL;
          a3 = v14 | *v87 & 7;
          *v87 = a3;
          *(_QWORD *)(v14 + 8) = v87;
          *v13 = a4 | *v13 & 7;
        }
      }
    }
    v9 = a1[4];
  }
  v15 = a1[6];
  if ( v15 != v9 )
  {
    if ( v7 )
    {
      a2 = (_BYTE *)a1[6];
      sub_2DB32C0(a1, a2, 1u, a4, a5, a6);
      v15 = a1[6];
    }
    v16 = a1[3];
    v17 = (unsigned __int64 *)sub_2E313E0(v15, a2, a3, a4, a5);
    v18 = a1[6];
    v19 = *(__int64 **)(v18 + 56);
    if ( v19 != (__int64 *)v17 )
    {
      v20 = (unsigned __int64 *)a1[92];
      if ( v17 != v20 )
      {
        a2 = (_BYTE *)(v18 + 40);
        sub_2E310C0(v16 + 40, v18 + 40, *(_QWORD *)(v18 + 56), v17);
        if ( v19 != (__int64 *)v17 )
        {
          a4 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v17;
          *v17 = *v17 & 7 | *v19 & 0xFFFFFFFFFFFFFFF8LL;
          v21 = *v20;
          *(_QWORD *)(a4 + 8) = v20;
          v21 &= 0xFFFFFFFFFFFFFFF8LL;
          a3 = v21 | *v19 & 7;
          *v19 = a3;
          *(_QWORD *)(v21 + 8) = v19;
          *v20 = a4 | *v20 & 7;
        }
      }
    }
    v9 = a1[4];
  }
  v22 = a1[3];
  v81 = *(_DWORD *)(v9 + 72);
  if ( v81 == 2 )
  {
    v53 = sub_2E313E0(v22, a2, a3, a4, a5);
    v54 = *(unsigned __int8 **)(v53 + 56);
    v55 = (_QWORD *)v53;
    v93 = v54;
    if ( v54 )
    {
      sub_B96E90((__int64)&v93, (__int64)v54, 1);
      v56 = (int *)a1[7];
      v57 = 8LL * *((unsigned int *)a1 + 16);
      v88 = &v56[v57];
      if ( v56 == &v56[v57] )
      {
LABEL_100:
        if ( v93 )
          sub_B91220((__int64)&v93, (__int64)v93);
        goto LABEL_36;
      }
    }
    else
    {
      v56 = (int *)a1[7];
      v69 = 8LL * *((unsigned int *)a1 + 16);
      v88 = &v56[v69];
      if ( &v56[v69] == v56 )
        goto LABEL_36;
    }
    v58 = v56;
    while ( 1 )
    {
      v92 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v58 + 32LL) + 8LL);
      if ( sub_2DB29F0(a1[2], *a1, v58[2], v58[3]) )
        break;
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, unsigned __int8 **, _QWORD, _QWORD, __int64, _QWORD, _QWORD))(*(_QWORD *)*a1 + 472LL))(
        *a1,
        a1[3],
        v55,
        &v93,
        v92,
        (unsigned int)v58[2],
        a1[41],
        *((unsigned int *)a1 + 84),
        (unsigned int)v58[3]);
LABEL_82:
      v68 = *(_QWORD *)v58;
      v58 += 8;
      sub_2E88E20(v68);
      *((_QWORD *)v58 - 4) = 0;
      if ( v88 == v58 )
        goto LABEL_100;
    }
    v59 = *(_QWORD *)(*a1 + 8);
    v94 = v93;
    v83 = v59 - 800;
    if ( v93 )
    {
      sub_B96E90((__int64)&v94, (__int64)v93, 1);
      v95 = v94;
      if ( v94 )
      {
        sub_B976B0((__int64)&v94, v94, (__int64)&v95);
        v60 = a1[3];
        v96 = 0;
        v97 = 0;
        v94 = 0;
        v61 = *(_QWORD *)(v60 + 32);
        v82 = v60;
        v98 = (unsigned __int64)v95;
        if ( v95 )
          sub_B96E90((__int64)&v98, (__int64)v95, 1);
        goto LABEL_72;
      }
    }
    else
    {
      v95 = 0;
    }
    v80 = a1[3];
    v96 = 0;
    v97 = 0;
    v82 = v80;
    v61 = *(_QWORD *)(v80 + 32);
    v98 = 0;
LABEL_72:
    v62 = sub_2E7B380(v61, v83, &v98, 0);
    v63 = (unsigned __int64 *)v62;
    if ( v98 )
    {
      v84 = v62;
      sub_B91220((__int64)&v98, v98);
      v63 = (unsigned __int64 *)v84;
    }
    v85 = v63;
    sub_2E31040(v82 + 40, v63);
    v64 = v85;
    v65 = *v85;
    v66 = *v55 & 0xFFFFFFFFFFFFFFF8LL;
    v85[1] = (unsigned __int64)v55;
    *v85 = v66 | v65 & 7;
    *(_QWORD *)(v66 + 8) = v85;
    *v55 = (unsigned __int64)v85 | *v55 & 7LL;
    if ( v96 )
    {
      sub_2E882B0(v85, v61);
      v64 = v85;
    }
    if ( v97 )
    {
      v86 = v64;
      sub_2E88680(v64, v61);
      v64 = v86;
    }
    v98 = 0x10000000;
    LODWORD(v99) = v92;
    v91 = v64;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    sub_2E8EAD0(v64, v61, &v98);
    v67 = v58[2];
    v98 = 0;
    v100 = 0;
    LODWORD(v99) = v67;
    v101 = 0;
    v102 = 0;
    sub_2E8EAD0(v91, v61, &v98);
    if ( v95 )
      sub_B91220((__int64)&v95, (__int64)v95);
    if ( v94 )
      sub_B91220((__int64)&v94, (__int64)v94);
    goto LABEL_82;
  }
  v23 = sub_2E313E0(v22, a2, a3, a4, a5);
  v24 = *(_QWORD *)(v23 + 56);
  v90 = v23;
  v98 = v24;
  if ( v24 )
  {
    sub_B96E90((__int64)&v98, v24, 1);
    v25 = (int *)a1[7];
    v26 = &v25[8 * *((unsigned int *)a1 + 16)];
    if ( v25 != v26 )
      goto LABEL_20;
LABEL_34:
    if ( v98 )
      sub_B91220((__int64)&v98, v98);
  }
  else
  {
    v25 = (int *)a1[7];
    v26 = &v25[8 * *((unsigned int *)a1 + 16)];
    if ( v26 != v25 )
    {
LABEL_20:
      v27 = v25;
      while ( 1 )
      {
        if ( sub_2DB29F0(a1[2], *a1, v27[2], v27[3]) )
        {
          LODWORD(v28) = v27[2];
        }
        else
        {
          v28 = (unsigned int)sub_2EC06C0(
                                a1[2],
                                *(_QWORD *)(*(_QWORD *)(a1[2] + 56)
                                          + 16LL * (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)v27 + 32LL) + 8LL) & 0x7FFFFFFF))
                              & 0xFFFFFFFFFFFFFFF8LL,
                                byte_3F871B3,
                                0);
          (*(void (__fastcall **)(__int64, __int64, __int64, unsigned __int64 *, __int64, _QWORD, __int64, _QWORD, _QWORD))(*(_QWORD *)*a1 + 472LL))(
            *a1,
            a1[3],
            v90,
            &v98,
            v28,
            (unsigned int)v27[2],
            a1[41],
            *((unsigned int *)a1 + 84),
            (unsigned int)v27[3]);
        }
        v29 = *(_QWORD *)v27;
        v30 = *(_DWORD *)(*(_QWORD *)v27 + 40LL) & 0xFFFFFF;
        if ( v30 != 1 )
          break;
LABEL_33:
        v27 += 8;
        if ( v26 == v27 )
          goto LABEL_34;
      }
      while ( 1 )
      {
        v32 = a1[4];
        v33 = v30 - 1;
        v34 = *(_QWORD *)(v29 + 32) + 40LL * (v30 - 1);
        v35 = a1[5];
        v36 = *(_QWORD *)(v34 + 24);
        if ( v35 == v32 )
        {
          v30 -= 2;
          if ( v36 != a1[3] )
          {
LABEL_26:
            v31 = a1[6];
            if ( v32 == v31 )
            {
              if ( v36 == a1[3] )
                goto LABEL_63;
            }
            else
            {
              if ( v36 != v31 )
                goto LABEL_28;
LABEL_63:
              sub_2E8A650(v29, v33);
              sub_2E8A650(*(_QWORD *)v27, v30);
            }
LABEL_28:
            if ( v30 == 1 )
              goto LABEL_33;
            goto LABEL_29;
          }
        }
        else
        {
          v30 -= 2;
          if ( v36 != v35 )
            goto LABEL_26;
        }
        *(_QWORD *)(v34 + 24) = a1[3];
        sub_2EAB0C0(*(_QWORD *)(*(_QWORD *)v27 + 32LL) + 40LL * v30, (unsigned int)v28);
        if ( v30 == 1 )
          goto LABEL_33;
LABEL_29:
        v29 = *(_QWORD *)v27;
      }
    }
  }
LABEL_36:
  sub_2E33650(a1[3], a1[5], 0);
  sub_2E33650(a1[3], a1[6], 1);
  v40 = a1[5];
  v41 = a1[4];
  if ( v40 != v41 )
  {
    sub_2E33650(v40, v41, 1);
    v41 = a1[4];
  }
  v42 = a1[6];
  if ( v42 != v41 )
    sub_2E33650(v42, v41, 1);
  v43 = *(unsigned __int8 **)(sub_2E313E0(a1[3], v41, v37, v38, v39) + 56);
  v95 = v43;
  if ( v43 )
    sub_B96E90((__int64)&v95, (__int64)v43, 1);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)*a1 + 360LL))(*a1, a1[3], 0);
  v46 = a1[5];
  v47 = a1[4];
  if ( v46 != v47 )
  {
    v48 = *(unsigned int *)(v89 + 8);
    if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(v89 + 12) )
    {
      sub_C8D5F0(v89, (const void *)(v89 + 16), v48 + 1, 8u, v44, v45);
      v48 = *(unsigned int *)(v89 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v89 + 8 * v48) = v46;
    ++*(_DWORD *)(v89 + 8);
    if ( a1[5] != (*(_QWORD *)(*(_QWORD *)(a1[5] + 32) + 320LL) & 0xFFFFFFFFFFFFFFF8LL) )
      sub_2E320F0();
    v47 = a1[4];
  }
  v49 = a1[6];
  if ( v49 != v47 )
  {
    v50 = *(unsigned int *)(v89 + 8);
    if ( v50 + 1 > (unsigned __int64)*(unsigned int *)(v89 + 12) )
    {
      sub_C8D5F0(v89, (const void *)(v89 + 16), v50 + 1, 8u, v44, v45);
      v50 = *(unsigned int *)(v89 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v89 + 8 * v50) = v49;
    ++*(_DWORD *)(v89 + 8);
    if ( a1[6] != (*(_QWORD *)(*(_QWORD *)(a1[6] + 32) + 320LL) & 0xFFFFFFFFFFFFFFF8LL) )
      sub_2E320F0();
    v47 = a1[4];
  }
  if ( v81 != 2 )
    goto LABEL_55;
  if ( !(unsigned __int8)sub_2E322F0(a1[3], v47) )
  {
    v47 = a1[4];
LABEL_55:
    v51 = *a1;
    v52 = a1[3];
    v98 = (unsigned __int64)&v100;
    v99 = 0;
    (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64 *, _QWORD, unsigned __int8 **, _QWORD))(*(_QWORD *)v51 + 368LL))(
      v51,
      v52,
      v47,
      0,
      &v100,
      0,
      &v95,
      0);
    sub_2E33F80(a1[3], a1[4], 0xFFFFFFFFLL);
    if ( (__int64 *)v98 != &v100 )
      _libc_free(v98);
    goto LABEL_57;
  }
  v70 = a1[4];
  v71 = a1[3];
  v72 = *(__int64 **)(v70 + 56);
  v73 = v70 + 48;
  if ( (__int64 *)(v70 + 48) != v72 && v73 != v71 + 48 )
  {
    sub_2E310C0(v71 + 40, v70 + 40, *(_QWORD *)(v70 + 56), v73);
    if ( (__int64 *)(v70 + 48) != v72 )
    {
      v74 = *(_QWORD *)(v70 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*v72 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v70 + 48;
      *(_QWORD *)(v70 + 48) = *(_QWORD *)(v70 + 48) & 7LL | *v72 & 0xFFFFFFFFFFFFFFF8LL;
      v75 = *(_QWORD *)(v71 + 48);
      *(_QWORD *)(v74 + 8) = v71 + 48;
      v75 &= 0xFFFFFFFFFFFFFFF8LL;
      *v72 = v75 | *v72 & 7;
      *(_QWORD *)(v75 + 8) = v72;
      *(_QWORD *)(v71 + 48) = v74 | *(_QWORD *)(v71 + 48) & 7LL;
    }
    v71 = a1[3];
    v70 = a1[4];
  }
  sub_2E34140(v71, v70);
  v78 = a1[4];
  v79 = *(unsigned int *)(v89 + 8);
  if ( v79 + 1 > (unsigned __int64)*(unsigned int *)(v89 + 12) )
  {
    sub_C8D5F0(v89, (const void *)(v89 + 16), v79 + 1, 8u, v76, v77);
    v79 = *(unsigned int *)(v89 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v89 + 8 * v79) = v78;
  ++*(_DWORD *)(v89 + 8);
  if ( a1[4] != (*(_QWORD *)(*(_QWORD *)(a1[4] + 32) + 320LL) & 0xFFFFFFFFFFFFFFF8LL) )
    sub_2E320F0();
LABEL_57:
  if ( v95 )
    sub_B91220((__int64)&v95, (__int64)v95);
}
