// Function: sub_148F0C0
// Address: 0x148f0c0
//
__int64 *__fastcall sub_148F0C0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5, __m128i a6, __m128i a7)
{
  __int64 v8; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v15; // eax
  __int64 v16; // r9
  __int16 v17; // ax
  _QWORD *v18; // r9
  _QWORD *v19; // r14
  __int64 v20; // rdi
  __int64 *v21; // rdi
  __int64 *v22; // rdi
  __int64 v23; // rax
  __int64 *v24; // rbx
  char v25; // r15
  bool v26; // al
  __int64 v27; // rax
  bool v28; // r8
  __int64 v29; // r14
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r9
  _QWORD *v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned int v40; // esi
  __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rbx
  char v44; // r9
  __int64 *v45; // rax
  int v46; // eax
  unsigned int v47; // eax
  __int64 v48; // rax
  char v49; // r8
  __int64 *v50; // rax
  int v51; // esi
  __int64 v52; // rdx
  char v53; // r8
  __int64 *v54; // rax
  int v55; // esi
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // r9
  __int64 *v59; // [rsp+10h] [rbp-110h]
  char v60; // [rsp+30h] [rbp-F0h]
  bool v61; // [rsp+30h] [rbp-F0h]
  __int64 v62; // [rsp+30h] [rbp-F0h]
  __int64 *v63; // [rsp+38h] [rbp-E8h]
  __int64 v64; // [rsp+40h] [rbp-E0h]
  __int64 v65; // [rsp+40h] [rbp-E0h]
  _QWORD *v66; // [rsp+40h] [rbp-E0h]
  __int64 v67; // [rsp+40h] [rbp-E0h]
  __int64 v68; // [rsp+48h] [rbp-D8h]
  _QWORD *v69; // [rsp+48h] [rbp-D8h]
  __int64 v70; // [rsp+48h] [rbp-D8h]
  int v71; // [rsp+48h] [rbp-D8h]
  __int64 v73; // [rsp+58h] [rbp-C8h]
  __int64 v74; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v75; // [rsp+68h] [rbp-B8h]
  __int64 *v76; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v77; // [rsp+78h] [rbp-A8h]
  _BYTE v78[16]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 *v79; // [rsp+90h] [rbp-90h] BYREF
  __int64 v80; // [rsp+98h] [rbp-88h]
  __int64 v81; // [rsp+A0h] [rbp-80h] BYREF
  unsigned int v82; // [rsp+A8h] [rbp-78h]
  __int64 *v83; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v84; // [rsp+B8h] [rbp-68h]
  __int64 v85; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v86; // [rsp+C8h] [rbp-58h]
  int v87; // [rsp+D0h] [rbp-50h]
  __int64 **v88; // [rsp+D8h] [rbp-48h]
  char v89; // [rsp+E0h] [rbp-40h]

  v8 = a2;
  v11 = sub_1456040(a3);
  v73 = sub_145CF80((__int64)a1, v11, 0, 0);
  v12 = sub_1456040(a3);
  v13 = sub_145CF80((__int64)a1, v12, 1, 0);
  v68 = v13;
  if ( a2 == a3 )
  {
    *a4 = v13;
    *a5 = v73;
    return a5;
  }
  if ( sub_14560B0(a2) )
  {
    *a4 = v73;
    *a5 = v73;
    return (__int64 *)v73;
  }
  LOBYTE(v15) = sub_1456110(a3);
  v16 = v15;
  if ( (_BYTE)v15 )
  {
    *a4 = a2;
    *a5 = v73;
    return a5;
  }
  v17 = *(_WORD *)(a3 + 24);
  if ( v17 != 5 )
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0:
        if ( !v17 )
        {
          v37 = *(_QWORD *)(a2 + 32);
          v75 = *(_DWORD *)(v37 + 32);
          if ( v75 > 0x40 )
            sub_16A4FD0(&v74, v37 + 24);
          else
            v74 = *(_QWORD *)(v37 + 24);
          v38 = *(_QWORD *)(a3 + 32);
          v39 = *(unsigned int *)(v38 + 32);
          LODWORD(v77) = v39;
          if ( (unsigned int)v39 > 0x40 )
          {
            sub_16A4FD0(&v76, v38 + 24);
            v39 = (unsigned int)v77;
          }
          else
          {
            v76 = *(__int64 **)(v38 + 24);
          }
          v40 = v75;
          if ( v75 > (unsigned int)v39 )
          {
            sub_16A5B10(&v83, &v76, v75);
            if ( (unsigned int)v77 > 0x40 && v76 )
              j_j___libc_free_0_0(v76);
            v76 = v83;
            v46 = v84;
            LODWORD(v84) = 0;
            LODWORD(v77) = v46;
            sub_135E100((__int64 *)&v83);
            v40 = v75;
          }
          else if ( v75 < (unsigned int)v39 )
          {
            sub_16A5B10(&v83, &v74, v39);
            if ( v75 > 0x40 )
            {
              if ( v74 )
                j_j___libc_free_0_0(v74);
            }
            v74 = (__int64)v83;
            v47 = v84;
            LODWORD(v84) = 0;
            v75 = v47;
            sub_135E100((__int64 *)&v83);
            v40 = v75;
          }
          sub_135E0D0((__int64)&v79, v40, 0, 0);
          sub_135E0D0((__int64)&v83, v75, 0, 0);
          sub_16AE5C0(&v74, &v76, &v79, &v83);
          v73 = sub_145CF40((__int64)a1, (__int64)&v79);
          v8 = sub_145CF40((__int64)a1, (__int64)&v83);
          sub_135E100((__int64 *)&v83);
          sub_135E100((__int64 *)&v79);
          sub_135E100((__int64 *)&v76);
          sub_135E100(&v74);
        }
        goto LABEL_18;
      case 1:
      case 2:
      case 3:
      case 6:
      case 8:
      case 9:
      case 0xA:
      case 0xB:
        goto LABEL_18;
      case 4:
        v79 = &v81;
        v83 = &v85;
        v80 = 0x200000000LL;
        v84 = 0x200000000LL;
        v32 = sub_1456040(a3);
        v34 = *(_QWORD **)(a2 + 32);
        v70 = v32;
        v66 = &v34[*(_QWORD *)(a2 + 40)];
        if ( v34 != v66 )
        {
          while ( 1 )
          {
            sub_148F0C0(a1, *v34, a3, &v74, &v76, v33);
            if ( v70 != sub_1456040(v74) || v70 != sub_1456040((__int64)v76) )
              break;
            v36 = (unsigned int)v80;
            if ( (unsigned int)v80 >= HIDWORD(v80) )
            {
              sub_16CD150(&v79, &v81, 0, 8);
              v36 = (unsigned int)v80;
            }
            v79[v36] = v74;
            v35 = (unsigned int)v84;
            LODWORD(v80) = v80 + 1;
            if ( (unsigned int)v84 >= HIDWORD(v84) )
            {
              sub_16CD150(&v83, &v85, 0, 8);
              v35 = (unsigned int)v84;
            }
            ++v34;
            v83[v35] = (__int64)v76;
            LODWORD(v84) = v84 + 1;
            if ( v66 == v34 )
              goto LABEL_64;
          }
          v8 = a2;
          v21 = v83;
LABEL_14:
          if ( v21 == &v85 )
            goto LABEL_16;
          goto LABEL_15;
        }
LABEL_64:
        if ( (_DWORD)v80 == 1 )
        {
          v21 = v83;
          v8 = *v83;
          v73 = *v79;
          goto LABEL_14;
        }
        v73 = (__int64)sub_147DD40((__int64)a1, (__int64 *)&v79, 0, 0, a6, a7);
        v45 = sub_147DD40((__int64)a1, (__int64 *)&v83, 0, 0, a6, a7);
        v21 = v83;
        v8 = (__int64)v45;
        if ( v83 != &v85 )
LABEL_15:
          _libc_free((unsigned __int64)v21);
LABEL_16:
        v22 = v79;
        if ( v79 != &v81 )
          goto LABEL_17;
        goto LABEL_18;
      case 5:
        v60 = v16;
        v76 = (__int64 *)v78;
        v77 = 0x200000000LL;
        v23 = sub_1456040(a3);
        v24 = *(__int64 **)(a2 + 32);
        v65 = v23;
        v63 = &v24[*(_QWORD *)(a2 + 40)];
        if ( v24 == v63 )
          goto LABEL_62;
        v59 = a4;
        v25 = v60;
        do
        {
          v29 = *v24;
          if ( v65 != sub_1456040(*v24) )
          {
LABEL_58:
            v8 = a2;
            a4 = v59;
            v22 = v76;
            goto LABEL_59;
          }
          if ( v25 || (sub_148F0C0(a1, v29, a3, &v79, &v83, v30), !(v26 = sub_14560B0((__int64)v83))) )
          {
            v31 = (unsigned int)v77;
            if ( (unsigned int)v77 >= HIDWORD(v77) )
            {
              sub_16CD150(&v76, v78, 0, 8);
              v31 = (unsigned int)v77;
            }
            v76[v31] = v29;
            LODWORD(v77) = v77 + 1;
          }
          else
          {
            v61 = v26;
            if ( v65 != sub_1456040((__int64)v79) )
              goto LABEL_58;
            v27 = (unsigned int)v77;
            v28 = v61;
            if ( (unsigned int)v77 >= HIDWORD(v77) )
            {
              sub_16CD150(&v76, v78, 0, 8);
              v27 = (unsigned int)v77;
              v28 = v61;
            }
            v25 = v28;
            v76[v27] = (__int64)v79;
            LODWORD(v77) = v77 + 1;
          }
          ++v24;
        }
        while ( v63 != v24 );
        v44 = v25;
        v8 = a2;
        a4 = v59;
        if ( v44 )
        {
          if ( (_DWORD)v77 == 1 )
          {
            v22 = v76;
            v8 = v73;
            v73 = *v76;
          }
          else
          {
            v57 = sub_147EE30(a1, &v76, 0, 0, a6, a7);
            v8 = v73;
            v22 = v76;
            v73 = v57;
          }
LABEL_59:
          if ( v22 != (__int64 *)v78 )
LABEL_17:
            _libc_free((unsigned __int64)v22);
LABEL_18:
          *a4 = v73;
          *a5 = v8;
          return a5;
        }
LABEL_62:
        if ( *(_WORD *)(a3 + 24) != 10 )
          goto LABEL_63;
        v48 = *(_QWORD *)(a3 - 8);
        v79 = 0;
        v81 = 0;
        v74 = v48;
        v82 = 0;
        v80 = 0;
        v49 = sub_145CB40((__int64)&v79, &v74, &v83);
        v50 = v83;
        if ( v49 )
          goto LABEL_86;
        v51 = v82;
        v79 = (__int64 *)((char *)v79 + 1);
        if ( 4 * ((int)v81 + 1) >= 3 * v82 )
        {
          v51 = 2 * v82;
        }
        else if ( v82 - HIDWORD(v81) - ((_DWORD)v81 + 1) > v82 >> 3 )
        {
LABEL_83:
          LODWORD(v81) = v81 + 1;
          if ( *v50 != -8 )
            --HIDWORD(v81);
          v52 = v74;
          v50[1] = 0;
          *v50 = v52;
LABEL_86:
          v50[1] = *(_QWORD *)(v73 + 32);
          v83 = a1;
          v84 = 0;
          v85 = 0;
          v86 = 0;
          v87 = 0;
          v88 = &v79;
          v89 = 1;
          v62 = sub_148F000((__int64)&v83, v8, a6, a7);
          j___libc_free_0(v85);
          if ( !sub_14560B0(v62) )
          {
            v67 = sub_14806B0((__int64)a1, v8, v62, 0, 0);
            v71 = sub_1458970(v67);
            if ( v71 <= (int)sub_1458970(v8) )
            {
              sub_148F0C0(a1, v67, a3, &v74, &v83, v58);
              if ( (__int64 *)v73 == v83 )
              {
                v73 = v74;
                j___libc_free_0(v80);
                if ( v76 != (__int64 *)v78 )
                  _libc_free((unsigned __int64)v76);
                v8 = v62;
                goto LABEL_18;
              }
            }
            goto LABEL_94;
          }
          v74 = *(_QWORD *)(a3 - 8);
          v53 = sub_145CB40((__int64)&v79, &v74, &v83);
          v54 = v83;
          if ( v53 )
          {
LABEL_93:
            v54[1] = *(_QWORD *)(v68 + 32);
            v83 = a1;
            v84 = 0;
            v85 = 0;
            v86 = 0;
            v87 = 0;
            v88 = &v79;
            v89 = 1;
            v73 = sub_148F000((__int64)&v83, v8, a6, a7);
            j___libc_free_0(v85);
            v8 = v62;
LABEL_94:
            j___libc_free_0(v80);
LABEL_63:
            v22 = v76;
            goto LABEL_59;
          }
          v55 = v82;
          v79 = (__int64 *)((char *)v79 + 1);
          if ( 4 * ((int)v81 + 1) >= 3 * v82 )
          {
            v55 = 2 * v82;
          }
          else if ( v82 - HIDWORD(v81) - ((_DWORD)v81 + 1) > v82 >> 3 )
          {
LABEL_90:
            LODWORD(v81) = v81 + 1;
            if ( *v54 != -8 )
              --HIDWORD(v81);
            v56 = v74;
            v54[1] = 0;
            *v54 = v56;
            goto LABEL_93;
          }
          sub_14669A0((__int64)&v79, v55);
          sub_145CB40((__int64)&v79, &v74, &v83);
          v54 = v83;
          goto LABEL_90;
        }
        sub_14669A0((__int64)&v79, v51);
        sub_145CB40((__int64)&v79, &v74, &v83);
        v50 = v83;
        goto LABEL_83;
      case 7:
        if ( *(_QWORD *)(a2 + 40) == 2 )
        {
          sub_148F0C0(a1, **(_QWORD **)(a2 + 32), a3, &v74, &v76, v16);
          v41 = sub_13A5BC0((_QWORD *)a2, (__int64)a1);
          sub_148F0C0(a1, v41, a3, &v79, &v83, v42);
          v43 = sub_1456040(a3);
          if ( v43 == sub_1456040(v74)
            && v43 == sub_1456040((__int64)v76)
            && v43 == sub_1456040((__int64)v79)
            && v43 == sub_1456040((__int64)v83) )
          {
            v73 = sub_14799E0((__int64)a1, v74, (__int64)v79, *(_QWORD *)(a2 + 48), *(_WORD *)(a2 + 26) & 7);
            v8 = sub_14799E0((__int64)a1, (__int64)v76, (__int64)v83, *(_QWORD *)(a2 + 48), *(_WORD *)(a2 + 26) & 7);
          }
        }
        goto LABEL_18;
    }
  }
  *a4 = a2;
  v18 = *(_QWORD **)(a3 + 32);
  v69 = &v18[*(_QWORD *)(a3 + 40)];
  if ( v18 == v69 )
  {
LABEL_20:
    *a5 = v73;
    return a5;
  }
  else
  {
    v64 = a2;
    v19 = *(_QWORD **)(a3 + 32);
    while ( 1 )
    {
      sub_148F0C0(a1, a2, *v19, &v79, &v83, v18);
      v20 = (__int64)v83;
      *a4 = (__int64)v79;
      if ( !sub_14560B0(v20) )
        break;
      if ( v69 == ++v19 )
        goto LABEL_20;
      a2 = *a4;
    }
    *a4 = v73;
    *a5 = v64;
    return a5;
  }
}
