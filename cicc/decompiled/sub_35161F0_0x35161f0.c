// Function: sub_35161F0
// Address: 0x35161f0
//
__int64 __fastcall sub_35161F0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, _QWORD **a5, __int64 a6)
{
  unsigned int v9; // ebx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned __int64 v13; // r12
  __int64 v14; // r15
  __int64 *v15; // rbx
  unsigned int v16; // r12d
  __int64 v17; // r14
  unsigned int v18; // eax
  char v19; // al
  __int64 v20; // r14
  unsigned __int64 *v21; // rcx
  unsigned __int64 *v22; // r12
  _QWORD *v23; // rdi
  _QWORD *v24; // rsi
  __int64 v25; // rsi
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  int v28; // eax
  __int64 v29; // r8
  int v30; // edx
  unsigned int v31; // eax
  int v32; // r9d
  unsigned int v33; // r13d
  unsigned __int64 v34; // rax
  unsigned __int64 *v35; // rdi
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rax
  bool v38; // cf
  unsigned __int64 v39; // rbx
  unsigned __int64 *v40; // rdi
  unsigned __int64 v41; // r12
  unsigned __int64 v42; // rax
  __int64 v43; // r12
  unsigned __int64 v44; // r13
  unsigned int v46; // r12d
  unsigned int v47; // r15d
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // r10
  unsigned __int64 *v51; // rdi
  unsigned __int64 v52; // rax
  unsigned __int64 *v53; // rdi
  unsigned __int64 v54; // rbx
  unsigned __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  unsigned int v58; // esi
  unsigned __int64 v59; // rbx
  unsigned __int64 v60; // r12
  unsigned __int64 v61; // rax
  __int64 *v62; // rax
  unsigned __int64 *v63; // rdi
  unsigned __int64 v64; // rax
  unsigned __int64 *v65; // rdi
  unsigned __int64 v66; // rbx
  unsigned __int64 v67; // r13
  unsigned __int64 v68; // rax
  __int64 v69; // rax
  unsigned __int64 v70; // rbx
  unsigned int v71; // [rsp+4h] [rbp-FCh]
  unsigned __int64 v73; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v74; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v75; // [rsp+20h] [rbp-E0h]
  unsigned int v76; // [rsp+38h] [rbp-C8h]
  unsigned int v77; // [rsp+3Ch] [rbp-C4h]
  unsigned __int64 v80; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v81; // [rsp+50h] [rbp-B0h]
  __int64 *v82; // [rsp+58h] [rbp-A8h]
  unsigned __int64 *v83; // [rsp+58h] [rbp-A8h]
  unsigned __int64 v84; // [rsp+58h] [rbp-A8h]
  unsigned int v85; // [rsp+6Ch] [rbp-94h] BYREF
  __int64 v86; // [rsp+70h] [rbp-90h] BYREF
  __int64 v87; // [rsp+78h] [rbp-88h] BYREF
  unsigned __int64 v88; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int64 v89; // [rsp+88h] [rbp-78h] BYREF
  unsigned __int64 v90; // [rsp+90h] [rbp-70h] BYREF
  __int64 v91; // [rsp+98h] [rbp-68h] BYREF
  __int64 *v92; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v93; // [rsp+A8h] [rbp-58h]
  _BYTE v94[80]; // [rsp+B0h] [rbp-50h] BYREF

  v92 = (__int64 *)v94;
  v86 = 0;
  v93 = 0x400000000LL;
  v76 = sub_3516000(a1, a3, a5, a6, (__int64)&v92);
  v9 = sub_2E441D0(*(_QWORD *)(a1 + 528), a2, a3);
  v10 = sub_2F06CB0(*(_QWORD *)(a1 + 536), a2);
  v11 = *(_QWORD *)(a1 + 536);
  v87 = v10;
  v88 = sub_2F06CB0(v11, a3);
  v75 = sub_1098D20((unsigned __int64 *)&v87, v9);
  v74 = sub_1098D20((unsigned __int64 *)&v87, a4);
  v12 = sub_2F06DD0(*(__int64 **)(a1 + 536));
  v13 = (unsigned int)v93;
  v73 = v12;
  if ( (_DWORD)v93 )
  {
    v14 = a1;
    v15 = v92;
    v16 = 0;
    v82 = &v92[(unsigned int)v93];
    do
    {
      v17 = *v15;
      v18 = sub_2E441D0(*(_QWORD *)(v14 + 528), a3, *v15);
      if ( v16 < v18 )
        v16 = v18;
      sub_2EB3EB0(*(_QWORD *)(v14 + 576), v17, a3);
      if ( v19 )
      {
        v86 = v17;
        v20 = v14;
        goto LABEL_9;
      }
      ++v15;
    }
    while ( v82 != v15 );
    v20 = v14;
LABEL_9:
    v21 = *(unsigned __int64 **)(a3 + 64);
    v83 = &v21[*(unsigned int *)(a3 + 72)];
    if ( v21 != v83 )
    {
      v71 = v16;
      v22 = *(unsigned __int64 **)(a3 + 64);
      v80 = 0;
      while ( 1 )
      {
        v27 = *v22;
        v90 = v27;
        if ( v27 != a3 && v27 != a2 && (_QWORD **)*sub_3515040(v20 + 888, (__int64 *)&v90) != a5 )
        {
          if ( !a6 )
          {
LABEL_12:
            v25 = v90;
LABEL_13:
            v77 = sub_2E441D0(*(_QWORD *)(v20 + 528), v25, a3);
            v91 = sub_2F06CB0(*(_QWORD *)(v20 + 536), v90);
            v26 = sub_1098D20((unsigned __int64 *)&v91, v77);
            if ( v80 >= v26 )
              v26 = v80;
            v80 = v26;
            goto LABEL_16;
          }
          if ( *(_DWORD *)(a6 + 16) )
          {
            v28 = *(_DWORD *)(a6 + 24);
            v29 = *(_QWORD *)(a6 + 8);
            if ( v28 )
            {
              v30 = v28 - 1;
              v31 = (v28 - 1) & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
              v25 = *(_QWORD *)(v29 + 8LL * v31);
              if ( v90 != v25 )
              {
                v32 = 1;
                while ( v25 != -4096 )
                {
                  v31 = v30 & (v32 + v31);
                  v25 = *(_QWORD *)(v29 + 8LL * v31);
                  if ( v90 == v25 )
                    goto LABEL_13;
                  ++v32;
                }
                goto LABEL_16;
              }
              goto LABEL_13;
            }
          }
          else
          {
            v23 = *(_QWORD **)(a6 + 32);
            v24 = &v23[*(unsigned int *)(a6 + 40)];
            if ( v24 != sub_3510810(v23, (__int64)v24, (__int64 *)&v90) )
              goto LABEL_12;
          }
        }
LABEL_16:
        if ( v83 == ++v22 )
        {
          v16 = v71;
          goto LABEL_29;
        }
      }
    }
    v80 = 0;
LABEL_29:
    v89 = v80;
    if ( !v86 )
      goto LABEL_32;
    if ( !sub_2E322C0(a3, v86) )
    {
      v80 = v89;
LABEL_32:
      v33 = v76 - v16;
      if ( v76 < v16 )
        v33 = 0;
      v34 = v88 - v80;
      if ( v88 <= v80 )
        v34 = 0;
      v90 = v34;
      v35 = &v89;
      v36 = sub_1098D20(&v88, v33);
      if ( v89 > v90 )
        v35 = &v90;
      v37 = sub_1098D20(v35, v16);
      v38 = __CFADD__(v75, v36);
      v39 = v75 + v36;
      v40 = &v89;
      v41 = v37;
      if ( v38 )
        v39 = -1;
      if ( v90 > v89 )
        v40 = &v90;
      v42 = sub_1098D20(v40, v33);
      v38 = __CFADD__(v74, v41);
      v43 = v74 + v41;
      if ( v38 )
        v43 = -1;
      v38 = __CFADD__(v42, v43);
      v13 = v42 + v43;
      v44 = v38;
      if ( v38 )
      {
        sub_F02DB0(&v85, qword_503C2C8, 0x64u);
        v44 = 0;
      }
      else
      {
        sub_F02DB0(&v85, qword_503C2C8, 0x64u);
        if ( v13 < v39 )
          v44 = v39 - v13;
      }
      v91 = v44;
      goto LABEL_48;
    }
    v46 = 0;
    v47 = sub_2E441D0(*(_QWORD *)(v20 + 528), a3, v86);
    if ( v76 >= v47 )
      v46 = v76 - v47;
    v81 = sub_1098D20(&v88, v47);
    v48 = sub_1098D20(&v88, v46);
    v49 = v89;
    v50 = v48;
    if ( v89 < v88 )
    {
      v61 = v88 - v89;
      v90 = v88 - v89;
      if ( v76 >> 1 >= v47 )
      {
LABEL_72:
        v51 = &v90;
        if ( v61 <= v49 )
          v51 = &v89;
LABEL_60:
        v52 = sub_1098D20(v51, v47);
        v53 = &v89;
        v54 = v52;
        if ( v89 > v90 )
          v53 = &v90;
        v55 = sub_1098D20(v53, v76);
        v56 = -1;
        v38 = __CFADD__(v74, v55);
        v57 = v74 + v55;
        if ( v38 )
          v57 = -1;
        v58 = qword_503C2C8;
        v38 = __CFADD__(v57, v54);
        v59 = v57 + v54;
        v13 = __CFADD__(v81, v75);
        if ( !v38 )
        {
          if ( !__CFADD__(v81, v75) )
            v56 = v81 + v75;
          v60 = v56;
          sub_F02DB0(&v85, qword_503C2C8, 0x64u);
          if ( v60 <= v59 )
            v13 = 0;
          else
            v13 = v60 - v59;
          goto LABEL_69;
        }
        if ( !__CFADD__(v81, v75) )
        {
          sub_F02DB0(&v85, qword_503C2C8, 0x64u);
LABEL_69:
          v91 = v13;
LABEL_48:
          LOBYTE(v13) = v73 <= sub_1098D70((unsigned __int64 *)&v91, v85);
          goto LABEL_49;
        }
LABEL_79:
        v13 = 0;
        sub_F02DB0(&v85, v58, 0x64u);
        goto LABEL_69;
      }
    }
    else
    {
      v90 = 0;
      v51 = &v89;
      if ( v76 >> 1 >= v47 )
        goto LABEL_60;
    }
    v84 = v50;
    v62 = sub_3515040(v20 + 888, &v86);
    if ( !(unsigned __int8)sub_35144C0(v20, a3, v86, *v62, v47, (__int64)a5, a6) )
    {
      v63 = &v89;
      if ( v89 > v90 )
        v63 = &v90;
      v64 = sub_1098D20(v63, v47);
      v65 = &v89;
      v66 = v64;
      if ( v90 > v89 )
        v65 = &v90;
      v67 = -1;
      v68 = sub_1098D20(v65, v46);
      v38 = __CFADD__(v74, v68);
      v69 = v74 + v68;
      v58 = qword_503C2C8;
      if ( v38 )
        v69 = -1;
      v38 = __CFADD__(v69, v66);
      v70 = v69 + v66;
      v13 = v38;
      if ( !v38 )
      {
        if ( !__CFADD__(v75, v84) )
          v67 = v75 + v84;
        sub_F02DB0(&v85, qword_503C2C8, 0x64u);
        if ( v67 > v70 )
          v13 = v67 - v70;
        goto LABEL_69;
      }
      goto LABEL_79;
    }
    v61 = v90;
    v49 = v89;
    goto LABEL_72;
  }
  sub_F02DB0(&v90, qword_503C2C8, 0x64u);
  if ( v74 < v75 )
    v13 = v75 - v74;
  v91 = v13;
  LOBYTE(v13) = v73 <= sub_1098D70((unsigned __int64 *)&v91, v90);
LABEL_49:
  if ( v92 != (__int64 *)v94 )
    _libc_free((unsigned __int64)v92);
  return (unsigned int)v13;
}
