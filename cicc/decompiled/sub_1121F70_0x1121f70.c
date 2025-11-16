// Function: sub_1121F70
// Address: 0x1121f70
//
void *__fastcall sub_1121F70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v5; // r14
  int v6; // ebx
  __int64 v8; // rdi
  __int64 v10; // r9
  int v11; // ebx
  __int64 v12; // r9
  __int64 v13; // rcx
  const void **v14; // r8
  __int64 v15; // r9
  const void **v16; // rcx
  const void **v17; // r8
  __int64 v18; // r9
  const void **v19; // rcx
  const void **v20; // r8
  bool v21; // al
  __int64 v22; // r9
  const void **v23; // rcx
  __int64 v24; // r8
  bool v25; // al
  __int64 v26; // rax
  void *result; // rax
  __int64 v28; // rdx
  _BYTE *v29; // rax
  _BYTE *v30; // rsi
  unsigned int **v31; // rdi
  _BYTE *v32; // rdx
  unsigned int **v33; // rdi
  __int64 v34; // rax
  __int16 v35; // r13
  __int64 v36; // r14
  __int16 v37; // r13
  __int64 v38; // r12
  bool v39; // al
  bool v40; // al
  unsigned int v41; // edx
  bool v42; // al
  unsigned int **v43; // r13
  _BYTE *v44; // rax
  __int64 v45; // r14
  __int16 v46; // r13
  __int64 v47; // r12
  bool v48; // al
  __int64 v49; // r9
  __int64 v50; // rcx
  __int64 v51; // r8
  int v52; // eax
  __int64 *v53; // rdi
  bool v54; // al
  __int64 v55; // rax
  __int64 *v56; // rdi
  unsigned int v57; // [rsp+0h] [rbp-B0h]
  const void **v58; // [rsp+8h] [rbp-A8h]
  const void **v59; // [rsp+8h] [rbp-A8h]
  __int64 v60; // [rsp+8h] [rbp-A8h]
  __int64 v61; // [rsp+8h] [rbp-A8h]
  __int64 v62; // [rsp+8h] [rbp-A8h]
  const void **v63; // [rsp+8h] [rbp-A8h]
  __int64 v64; // [rsp+10h] [rbp-A0h]
  const void **v65; // [rsp+10h] [rbp-A0h]
  const void **v66; // [rsp+10h] [rbp-A0h]
  const void **v67; // [rsp+10h] [rbp-A0h]
  const void **v68; // [rsp+10h] [rbp-A0h]
  __int64 v69; // [rsp+10h] [rbp-A0h]
  __int64 v70; // [rsp+10h] [rbp-A0h]
  const void **v71; // [rsp+10h] [rbp-A0h]
  __int64 v72; // [rsp+10h] [rbp-A0h]
  __int64 v73; // [rsp+10h] [rbp-A0h]
  const void **v74; // [rsp+10h] [rbp-A0h]
  const void **v75; // [rsp+10h] [rbp-A0h]
  __int64 v76; // [rsp+10h] [rbp-A0h]
  __int64 v77; // [rsp+18h] [rbp-98h]
  const void **v78; // [rsp+18h] [rbp-98h]
  const void **v79; // [rsp+18h] [rbp-98h]
  __int64 v80; // [rsp+18h] [rbp-98h]
  __int64 v81; // [rsp+18h] [rbp-98h]
  __int64 v82; // [rsp+18h] [rbp-98h]
  __int64 v83; // [rsp+18h] [rbp-98h]
  __int64 v84; // [rsp+18h] [rbp-98h]
  __int64 v85; // [rsp+18h] [rbp-98h]
  __int64 v86; // [rsp+18h] [rbp-98h]
  __int64 v87; // [rsp+18h] [rbp-98h]
  __int64 v88; // [rsp+18h] [rbp-98h]
  __int64 v89; // [rsp+18h] [rbp-98h]
  __int64 v90; // [rsp+20h] [rbp-90h]
  __int64 v91; // [rsp+20h] [rbp-90h]
  __int64 v92; // [rsp+20h] [rbp-90h]
  char v93; // [rsp+20h] [rbp-90h]
  __int64 v94; // [rsp+20h] [rbp-90h]
  __int64 v95; // [rsp+20h] [rbp-90h]
  __int64 v96; // [rsp+20h] [rbp-90h]
  __int64 v97; // [rsp+20h] [rbp-90h]
  const void **v98; // [rsp+20h] [rbp-90h]
  __int64 v99; // [rsp+20h] [rbp-90h]
  __int64 v100; // [rsp+20h] [rbp-90h]
  const void **v101; // [rsp+20h] [rbp-90h]
  unsigned __int64 v102; // [rsp+28h] [rbp-88h]
  bool v103; // [rsp+28h] [rbp-88h]
  __int64 v104; // [rsp+28h] [rbp-88h]
  __int64 v105; // [rsp+28h] [rbp-88h]
  __int64 v106; // [rsp+28h] [rbp-88h]
  void *v107; // [rsp+28h] [rbp-88h]
  const void **v108; // [rsp+28h] [rbp-88h]
  __int64 v109; // [rsp+28h] [rbp-88h]
  void *v110; // [rsp+28h] [rbp-88h]
  void *v111; // [rsp+28h] [rbp-88h]
  __int64 v112; // [rsp+28h] [rbp-88h]
  void *v113; // [rsp+28h] [rbp-88h]
  const void *v114; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v115; // [rsp+38h] [rbp-78h]
  const void *v116; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v117; // [rsp+48h] [rbp-68h]
  const void *v118; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v119; // [rsp+58h] [rbp-58h]
  __int16 v120; // [rsp+70h] [rbp-40h]

  v5 = *(unsigned __int8 **)(a3 - 64);
  v6 = *v5;
  if ( (unsigned __int8)(v6 - 54) > 2u )
    return 0;
  v8 = *((_QWORD *)v5 - 4);
  v10 = a3;
  v11 = v6 - 29;
  if ( *(_BYTE *)v8 == 17 )
  {
    v102 = v8 + 24;
  }
  else
  {
    v28 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
    if ( (unsigned int)v28 > 1
      || *(_BYTE *)v8 > 0x15u
      || (v83 = a5, v95 = a4, v105 = v10, v29 = sub_AD7630(v8, 0, v28), v10 = v105, a4 = v95, a5 = v83, !v29)
      || *v29 != 17 )
    {
LABEL_25:
      v26 = *((_QWORD *)v5 + 2);
      if ( v26 )
      {
        if ( !*(_QWORD *)(v26 + 8) )
        {
          v96 = a5;
          v106 = v10;
          if ( sub_9867B0(a4) && (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 1 && *v5 != 56 )
          {
            if ( v11 == 25 )
            {
              if ( **((_BYTE **)v5 - 8) > 0x15u )
              {
                v56 = *(__int64 **)(a1 + 32);
                v120 = 257;
                v32 = (_BYTE *)sub_F94560(v56, *(_QWORD *)(v106 - 32), *((_QWORD *)v5 - 4), (__int64)&v118, 0);
LABEL_41:
                v33 = *(unsigned int ***)(a1 + 32);
                v120 = 257;
                v34 = sub_A82350(v33, *((_BYTE **)v5 - 8), v32, (__int64)&v118);
                v35 = *(_WORD *)(a2 + 2);
                v36 = v34;
                v120 = 257;
                v37 = v35 & 0x3F;
                v38 = *(_QWORD *)(a2 - 32);
                result = sub_BD2C40(72, unk_3F10FD0);
                if ( result )
                {
                  v107 = result;
                  sub_1113300((__int64)result, v37, v36, v38, (__int64)&v118);
                  return v107;
                }
                return result;
              }
            }
            else if ( sub_D94040(v96) || **((_BYTE **)v5 - 8) > 0x15u )
            {
              v30 = *(_BYTE **)(v106 - 32);
              v31 = *(unsigned int ***)(a1 + 32);
              v120 = 257;
              v32 = (_BYTE *)sub_920A70(v31, v30, *((_BYTE **)v5 - 4), (__int64)&v118, 0, 0);
              goto LABEL_41;
            }
          }
        }
      }
      return 0;
    }
    v102 = (unsigned __int64)(v29 + 24);
  }
  v115 = 1;
  v114 = 0;
  v117 = 1;
  v116 = 0;
  if ( v11 == 25 )
  {
    v72 = a5;
    v85 = a4;
    v99 = v10;
    v48 = sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F);
    v49 = v99;
    v50 = v85;
    v51 = v72;
    if ( !v48 )
      goto LABEL_61;
    v52 = *(_DWORD *)(v72 + 8);
    v53 = (__int64 *)v72;
    v73 = v85;
    v86 = v99;
    v100 = v51;
    if ( !sub_986C60(v53, v52 - 1) )
    {
      v54 = sub_986C60((__int64 *)v73, *(_DWORD *)(v73 + 8) - 1);
      v50 = v73;
      v51 = v100;
      v49 = v86;
      if ( !v54 )
      {
LABEL_61:
        v62 = v49;
        v74 = (const void **)v50;
        v87 = v51;
        sub_9865C0((__int64)&v118, v50);
        sub_C48380((__int64)&v118, v102);
        sub_1110A30((__int64 *)&v116, (__int64 *)&v118);
        sub_969240((__int64 *)&v118);
        sub_9865C0((__int64)&v118, v87);
        sub_C48380((__int64)&v118, v102);
        sub_1110A30((__int64 *)&v114, (__int64 *)&v118);
        sub_969240((__int64 *)&v118);
        sub_9865C0((__int64)&v118, (__int64)&v116);
        sub_C47AC0((__int64)&v118, v102);
        v93 = sub_AAD8B0((__int64)&v118, v74) ^ 1;
        sub_969240((__int64 *)&v118);
        v23 = v74;
        v24 = v87;
        v22 = v62;
        goto LABEL_21;
      }
    }
LABEL_54:
    v41 = v117;
    result = 0;
    goto LABEL_48;
  }
  if ( v11 == 26 )
  {
    v60 = v10;
    v98 = (const void **)a4;
    v70 = a5;
    sub_9865C0((__int64)&v118, a4);
    sub_C47AC0((__int64)&v118, v102);
    sub_1110A30((__int64 *)&v116, (__int64 *)&v118);
    sub_969240((__int64 *)&v118);
    sub_9865C0((__int64)&v118, v70);
    sub_C47AC0((__int64)&v118, v102);
    sub_1110A30((__int64 *)&v114, (__int64 *)&v118);
    sub_969240((__int64 *)&v118);
    sub_9865C0((__int64)&v118, (__int64)&v116);
    sub_C48380((__int64)&v118, v102);
    v108 = v98;
    v93 = sub_AAD8B0((__int64)&v118, v98) ^ 1;
    sub_969240((__int64 *)&v118);
    v39 = sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F);
    v23 = v108;
    v24 = v70;
    v22 = v60;
    if ( v39 )
    {
      v61 = v70;
      v71 = v108;
      v109 = v22;
      v40 = sub_986C60((__int64 *)&v114, v115 - 1);
      v41 = v117;
      if ( v40
        || (v57 = v117, v42 = sub_986C60((__int64 *)&v116, v117 - 1), v41 = v57, v22 = v109, v23 = v71, v24 = v61, v42) )
      {
        result = 0;
        goto LABEL_48;
      }
    }
    goto LABEL_21;
  }
  v119 = *(_DWORD *)(a4 + 8);
  if ( v119 > 0x40 )
  {
    v69 = a5;
    v84 = v10;
    v97 = a4;
    sub_C43780((__int64)&v118, (const void **)a4);
    a5 = v69;
    v10 = v84;
    a4 = v97;
  }
  else
  {
    v118 = *(const void **)a4;
  }
  v64 = a5;
  v77 = a4;
  v90 = v10;
  sub_C47AC0((__int64)&v118, v102);
  v12 = v90;
  v13 = v77;
  v14 = (const void **)v64;
  v116 = v118;
  v117 = v119;
  v119 = *(_DWORD *)(v64 + 8);
  if ( v119 > 0x40 )
  {
    v76 = v77;
    v89 = v90;
    v101 = v14;
    sub_C43780((__int64)&v118, v14);
    v13 = v76;
    v12 = v89;
    v14 = v101;
  }
  else
  {
    v118 = *(const void **)v64;
  }
  v65 = v14;
  v78 = (const void **)v13;
  v91 = v12;
  sub_C47AC0((__int64)&v118, v102);
  v15 = v91;
  v16 = v78;
  v17 = v65;
  v114 = v118;
  v115 = v119;
  v119 = v117;
  if ( v117 > 0x40 )
  {
    sub_C43780((__int64)&v118, &v116);
    v17 = v65;
    v16 = v78;
    v15 = v91;
  }
  else
  {
    v118 = v116;
  }
  v66 = v17;
  v79 = v16;
  v92 = v15;
  sub_C44D10((__int64)&v118, v102);
  v18 = v92;
  v19 = v79;
  v20 = v66;
  if ( v119 <= 0x40 )
  {
    v93 = v118 != *v79;
  }
  else
  {
    v58 = v66;
    v21 = sub_C43C50((__int64)&v118, v79);
    v19 = v79;
    v18 = v92;
    v20 = v66;
    v93 = !v21;
    if ( v118 )
    {
      v67 = v79;
      v80 = v18;
      j_j___libc_free_0_0(v118);
      v20 = v58;
      v19 = v67;
      v18 = v80;
    }
  }
  v119 = v115;
  if ( v115 > 0x40 )
  {
    v63 = v20;
    v75 = v19;
    v88 = v18;
    sub_C43780((__int64)&v118, &v114);
    v20 = v63;
    v19 = v75;
    v18 = v88;
  }
  else
  {
    v118 = v114;
  }
  v59 = v20;
  v68 = v19;
  v81 = v18;
  sub_C44D10((__int64)&v118, v102);
  v22 = v81;
  v23 = v68;
  v24 = (__int64)v59;
  if ( v119 <= 0x40 )
  {
    v25 = v118 == *v59;
  }
  else
  {
    v25 = sub_C43C50((__int64)&v118, v59);
    v24 = (__int64)v59;
    v22 = v81;
    v23 = v68;
    if ( v118 )
    {
      v103 = v25;
      j_j___libc_free_0_0(v118);
      v24 = (__int64)v59;
      v23 = v68;
      v22 = v81;
      v25 = v103;
    }
  }
  if ( !v25 )
    goto LABEL_54;
LABEL_21:
  if ( v93 )
  {
    if ( (*(_WORD *)(a2 + 2) & 0x3F) == 0x20 )
    {
      v55 = sub_AD6450(*(_QWORD *)(a2 + 8));
    }
    else
    {
      if ( (*(_WORD *)(a2 + 2) & 0x3F) != 0x21 )
      {
        v82 = v24;
        v94 = (__int64)v23;
        v104 = v22;
        sub_969240((__int64 *)&v116);
        sub_969240((__int64 *)&v114);
        v10 = v104;
        a4 = v94;
        a5 = v82;
        goto LABEL_25;
      }
      v55 = sub_AD6400(*(_QWORD *)(a2 + 8));
    }
    result = sub_F162A0(a1, a2, v55);
  }
  else
  {
    v43 = *(unsigned int ***)(a1 + 32);
    v112 = v22;
    v120 = 257;
    v44 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v22 + 8), (__int64)&v114);
    v45 = sub_A82350(v43, *((_BYTE **)v5 - 8), v44, (__int64)&v118);
    v46 = *(_WORD *)(a2 + 2) & 0x3F;
    v47 = sub_AD8D80(*(_QWORD *)(v112 + 8), (__int64)&v116);
    v120 = 257;
    result = sub_BD2C40(72, unk_3F10FD0);
    if ( result )
    {
      v113 = result;
      sub_1113300((__int64)result, v46, v45, v47, (__int64)&v118);
      result = v113;
    }
  }
  v41 = v117;
LABEL_48:
  if ( v41 > 0x40 && v116 )
  {
    v110 = result;
    j_j___libc_free_0_0(v116);
    result = v110;
  }
  if ( v115 > 0x40 && v114 )
  {
    v111 = result;
    j_j___libc_free_0_0(v114);
    return v111;
  }
  return result;
}
