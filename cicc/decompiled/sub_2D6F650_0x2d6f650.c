// Function: sub_2D6F650
// Address: 0x2d6f650
//
__int64 __fastcall sub_2D6F650(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v5; // r13
  bool v7; // bl
  __int64 v8; // r15
  _QWORD *v9; // r14
  __int64 v10; // rbx
  _QWORD *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r10
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r10
  int *v19; // rbx
  int v20; // ebx
  bool v21; // zf
  __int64 v22; // rax
  int v23; // esi
  int v24; // edx
  unsigned int v25; // esi
  _QWORD *v26; // rbx
  int v27; // eax
  unsigned int v28; // esi
  int v29; // eax
  _QWORD *v30; // rdi
  _QWORD *v31; // rbx
  __int64 v32; // rdx
  __int64 v33; // r13
  __int64 v34; // rbx
  bool v35; // cl
  _QWORD *v36; // r15
  __int64 v37; // r12
  __int64 v38; // rax
  __int64 v39; // r9
  __int64 v40; // r14
  __int64 v41; // rax
  char v42; // al
  __int64 v43; // r9
  _QWORD *v44; // rax
  int *v45; // rax
  int v46; // eax
  unsigned int v47; // esi
  __int64 v48; // rdi
  __int64 v49; // r9
  int v50; // edx
  __int64 v51; // rax
  __int64 v52; // r14
  __int64 v54; // r8
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // r10
  int v58; // esi
  int v59; // edx
  unsigned int v60; // esi
  __int64 v61; // r13
  int v62; // eax
  __int64 v63; // rbx
  __int64 v64; // [rsp+8h] [rbp-108h]
  __int64 v67; // [rsp+20h] [rbp-F0h]
  __int64 v68; // [rsp+28h] [rbp-E8h]
  __int64 v69; // [rsp+40h] [rbp-D0h]
  __int64 v70; // [rsp+40h] [rbp-D0h]
  int v71; // [rsp+40h] [rbp-D0h]
  __int64 v72; // [rsp+48h] [rbp-C8h]
  __int64 v73; // [rsp+48h] [rbp-C8h]
  __int64 v75; // [rsp+50h] [rbp-C0h]
  __int64 v76; // [rsp+58h] [rbp-B8h]
  __int64 v77; // [rsp+58h] [rbp-B8h]
  __int64 v78; // [rsp+58h] [rbp-B8h]
  int v79; // [rsp+58h] [rbp-B8h]
  bool v80; // [rsp+58h] [rbp-B8h]
  bool v81; // [rsp+58h] [rbp-B8h]
  _QWORD *v82; // [rsp+58h] [rbp-B8h]
  __int64 v85; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v86; // [rsp+78h] [rbp-98h] BYREF
  _QWORD *v87; // [rsp+80h] [rbp-90h] BYREF
  __int64 v88; // [rsp+88h] [rbp-88h]
  __int64 v89; // [rsp+90h] [rbp-80h]
  __int64 v90; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v91; // [rsp+A8h] [rbp-68h]
  __int64 v92; // [rsp+B0h] [rbp-60h]
  unsigned __int64 v93; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v94; // [rsp+C8h] [rbp-48h]
  __int64 v95; // [rsp+D0h] [rbp-40h]
  __int64 v96; // [rsp+D8h] [rbp-38h]

  v76 = (a3 - 1) / 2;
  if ( a2 >= v76 )
  {
    v8 = a2;
    v31 = (_QWORD *)(a1 + 32 * a2);
    goto LABEL_44;
  }
  v5 = a2;
  v69 = a5 + 728;
  while ( 1 )
  {
    v8 = 2 * (v5 + 1);
    v13 = (v5 + 1) << 6;
    v14 = a1 + v13 - 32;
    v9 = (_QWORD *)(a1 + v13);
    v15 = *(_QWORD *)(v14 + 16);
    v10 = *(_QWORD *)(a1 + v13 + 16);
    v16 = *(_QWORD *)(v14 + 24);
    if ( v15 == v10 )
      goto LABEL_7;
    if ( v16 == v9[3] )
    {
      v92 = *(_QWORD *)(a1 + v13 + 16);
      v90 = 0;
      v91 = 0;
      if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
      {
        v72 = v15;
        sub_BD73F0((__int64)&v90);
        v15 = v72;
      }
      v73 = v15;
      v17 = sub_2D67BB0(v69, (__int64)&v90, &v86);
      v18 = v73;
      if ( v17 )
      {
        v19 = (int *)(v86 + 24);
        goto LABEL_23;
      }
      v26 = (_QWORD *)v86;
      v27 = *(_DWORD *)(a5 + 744);
      ++*(_QWORD *)(a5 + 728);
      v28 = *(_DWORD *)(a5 + 752);
      v87 = v26;
      v29 = v27 + 1;
      if ( 4 * v29 >= 3 * v28 )
      {
        v63 = v69;
        v64 = v73;
        sub_2D6E640(v69, 2 * v28);
      }
      else
      {
        if ( v28 - *(_DWORD *)(a5 + 748) - v29 > v28 >> 3 )
        {
LABEL_40:
          v93 = 0;
          v94 = 0;
          *(_DWORD *)(a5 + 744) = v29;
          v95 = -4096;
          if ( v26[2] != -4096 )
            --*(_DWORD *)(a5 + 748);
          v67 = v18;
          sub_D68D70(&v93);
          v30 = v26;
          v19 = (int *)(v26 + 3);
          sub_2D57220(v30, v92);
          *v19 = 0;
          v18 = v67;
LABEL_23:
          v20 = *v19;
          v89 = v18;
          v87 = 0;
          v88 = 0;
          if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
            sub_BD73F0((__int64)&v87);
          v21 = (unsigned __int8)sub_2D67BB0(v69, (__int64)&v87, &v85) == 0;
          v22 = v85;
          if ( !v21 )
            goto LABEL_32;
          v86 = v85;
          v23 = *(_DWORD *)(a5 + 744);
          ++*(_QWORD *)(a5 + 728);
          v24 = v23 + 1;
          v25 = *(_DWORD *)(a5 + 752);
          if ( 4 * v24 >= 3 * v25 )
          {
            v25 *= 2;
          }
          else if ( v25 - *(_DWORD *)(a5 + 748) - v24 > v25 >> 3 )
          {
LABEL_29:
            v93 = 0;
            v94 = 0;
            *(_DWORD *)(a5 + 744) = v24;
            v95 = -4096;
            if ( *(_QWORD *)(v22 + 16) != -4096 )
              --*(_DWORD *)(a5 + 748);
            v68 = v22;
            sub_D68D70(&v93);
            sub_2D57220((_QWORD *)v68, v89);
            v22 = v68;
            *(_DWORD *)(v68 + 24) = 0;
LABEL_32:
            v7 = v20 < *(_DWORD *)(v22 + 24);
            if ( v89 != 0 && v89 != -4096 && v89 != -8192 )
              sub_BD60C0(&v87);
            if ( v92 != 0 && v92 != -4096 && v92 != -8192 )
              sub_BD60C0(&v90);
            goto LABEL_4;
          }
          sub_2D6E640(v69, v25);
          sub_2D67BB0(v69, (__int64)&v87, &v86);
          v24 = *(_DWORD *)(a5 + 744) + 1;
          v22 = v86;
          goto LABEL_29;
        }
        v63 = v69;
        v64 = v73;
        sub_2D6E640(v69, v28);
      }
      sub_2D67BB0(v63, (__int64)&v90, &v87);
      v26 = v87;
      v18 = v64;
      v29 = *(_DWORD *)(a5 + 744) + 1;
      goto LABEL_40;
    }
    v7 = v16 > v9[3];
LABEL_4:
    if ( v7 )
    {
      --v8;
      v9 = (_QWORD *)(a1 + 32 * v8);
    }
    v10 = v9[2];
LABEL_7:
    v11 = (_QWORD *)(a1 + 32 * v5);
    v12 = v11[2];
    if ( v12 != v10 )
    {
      if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
        sub_BD60C0(v11);
      v11[2] = v10;
      if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
        sub_BD73F0((__int64)v11);
    }
    v11[3] = v9[3];
    if ( v8 >= v76 )
      break;
    v5 = v8;
  }
  v31 = v9;
LABEL_44:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v8 )
  {
    v61 = a1 + ((v8 + 1) << 6) - 32;
    v8 = 2 * (v8 + 1) - 1;
    sub_2D57220(v31, *(_QWORD *)(v61 + 16));
    v31[3] = *(_QWORD *)(v61 + 24);
    v31 = (_QWORD *)(a1 + 32 * v8);
  }
  sub_D68CD0(&v93, 0, a4);
  v32 = a4[3];
  v96 = v32;
  if ( v8 <= a2 )
  {
    v52 = v95;
    goto LABEL_78;
  }
  v33 = a5;
  v34 = (v8 - 1) / 2;
  v75 = a5 + 728;
  while ( 2 )
  {
    v39 = v95;
    v40 = a1 + 32 * v34;
    v41 = *(_QWORD *)(v40 + 16);
    if ( v95 == v41 )
    {
      v52 = v95;
      v31 = (_QWORD *)(a1 + 32 * v8);
      goto LABEL_78;
    }
    if ( *(_QWORD *)(v40 + 24) != v32 )
    {
      v35 = *(_QWORD *)(v40 + 24) < v32;
LABEL_50:
      v36 = (_QWORD *)(a1 + 32 * v8);
      if ( !v35 )
        goto LABEL_87;
      goto LABEL_51;
    }
    v90 = 0;
    v91 = 0;
    v92 = v41;
    if ( v41 != 0 && v41 != -4096 && v41 != -8192 )
    {
      v77 = v95;
      sub_BD73F0((__int64)&v90);
      v39 = v77;
    }
    v78 = v39;
    v42 = sub_2D67BB0(v75, (__int64)&v90, &v86);
    v43 = v78;
    v21 = v42 == 0;
    v44 = (_QWORD *)v86;
    if ( !v21 )
    {
      v45 = (int *)(v86 + 24);
      goto LABEL_67;
    }
    v58 = *(_DWORD *)(v33 + 744);
    ++*(_QWORD *)(v33 + 728);
    v87 = v44;
    v59 = v58 + 1;
    v60 = *(_DWORD *)(v33 + 752);
    if ( 4 * v59 >= 3 * v60 )
    {
      v60 *= 2;
LABEL_97:
      sub_2D6E640(v75, v60);
      sub_2D67BB0(v75, (__int64)&v90, &v87);
      v44 = v87;
      v43 = v78;
      goto LABEL_90;
    }
    if ( v60 - *(_DWORD *)(v33 + 748) - v59 <= v60 >> 3 )
      goto LABEL_97;
LABEL_90:
    ++*(_DWORD *)(v33 + 744);
    if ( v44[2] != -4096 )
      --*(_DWORD *)(v33 + 748);
    v70 = v43;
    v82 = v44;
    sub_2D57220(v44, v92);
    v43 = v70;
    *((_DWORD *)v82 + 6) = 0;
    v45 = (int *)(v82 + 3);
LABEL_67:
    v46 = *v45;
    v89 = v43;
    v87 = 0;
    v79 = v46;
    v88 = 0;
    if ( v43 != 0 && v43 != -4096 && v43 != -8192 )
      sub_BD73F0((__int64)&v87);
    v47 = *(_DWORD *)(v33 + 752);
    if ( !v47 )
    {
      ++*(_QWORD *)(v33 + 728);
      v86 = 0;
      goto LABEL_72;
    }
    v49 = v89;
    v54 = *(_QWORD *)(v33 + 736);
    v51 = v89;
    v55 = (v47 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
    v56 = v54 + 32 * v55;
    v57 = *(_QWORD *)(v56 + 16);
    if ( v57 == v89 )
    {
LABEL_80:
      v50 = *(_DWORD *)(v56 + 24);
    }
    else
    {
      v71 = 1;
      v48 = 0;
      while ( v57 != -4096 )
      {
        if ( v57 == -8192 && !v48 )
          v48 = v56;
        LODWORD(v55) = (v47 - 1) & (v71 + v55);
        v56 = v54 + 32LL * (unsigned int)v55;
        v57 = *(_QWORD *)(v56 + 16);
        if ( v89 == v57 )
        {
          v51 = v89;
          goto LABEL_80;
        }
        ++v71;
      }
      v62 = *(_DWORD *)(v33 + 744);
      if ( !v48 )
        v48 = v56;
      ++*(_QWORD *)(v33 + 728);
      v86 = v48;
      if ( 4 * (v62 + 1) >= 3 * v47 )
      {
LABEL_72:
        v47 *= 2;
      }
      else if ( v47 - *(_DWORD *)(v33 + 748) - (v62 + 1) > v47 >> 3 )
      {
        goto LABEL_74;
      }
      sub_2D6E640(v75, v47);
      sub_2D67BB0(v75, (__int64)&v87, &v86);
      v48 = v86;
      v49 = v89;
LABEL_74:
      ++*(_DWORD *)(v33 + 744);
      if ( *(_QWORD *)(v48 + 16) != -4096 )
        --*(_DWORD *)(v33 + 748);
      sub_2D57220((_QWORD *)v48, v49);
      v50 = 0;
      *(_DWORD *)(v48 + 24) = 0;
      v51 = v89;
    }
    v35 = v79 < v50;
    if ( v51 != 0 && v51 != -4096 && v51 != -8192 )
    {
      v80 = v79 < v50;
      sub_BD60C0(&v87);
      v35 = v80;
    }
    if ( v92 == 0 || v92 == -4096 || v92 == -8192 )
      goto LABEL_50;
    v81 = v35;
    sub_BD60C0(&v90);
    v36 = (_QWORD *)(a1 + 32 * v8);
    if ( !v81 )
    {
LABEL_87:
      v52 = v95;
      v31 = v36;
      goto LABEL_78;
    }
LABEL_51:
    v37 = *(_QWORD *)(v40 + 16);
    v38 = v36[2];
    if ( v37 != v38 )
    {
      if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
        sub_BD60C0(v36);
      v36[2] = v37;
      if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
        sub_BD73F0((__int64)v36);
    }
    v36[3] = *(_QWORD *)(v40 + 24);
    if ( a2 < v34 )
    {
      v32 = v96;
      v8 = v34;
      v34 = (v34 - 1) / 2;
      continue;
    }
    break;
  }
  v31 = (_QWORD *)(a1 + 32 * v34);
  v52 = v95;
LABEL_78:
  sub_2D57220(v31, v52);
  v31[3] = v96;
  return sub_D68D70(&v93);
}
