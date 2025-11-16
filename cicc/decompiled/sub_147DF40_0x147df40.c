// Function: sub_147DF40
// Address: 0x147df40
//
__int64 __fastcall sub_147DF40(__int64 a1, unsigned int *a2, __int64 *a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v16; // rsi
  __int16 v17; // ax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rsi
  bool v21; // al
  __int64 v22; // r15
  __int64 *v23; // rax
  __int64 v24; // rdx
  bool v25; // al
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  char v29; // al
  unsigned int v30; // eax
  __int64 *v31; // rax
  __int64 v32; // r15
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // rax
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // rax
  bool v40; // r15
  __int64 v41; // rax
  __int64 *v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rcx
  unsigned int v45; // r8d
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned int v49; // eax
  __int64 *v50; // rax
  __int64 v51; // r15
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 *v54; // rsi
  __int64 v55; // rdx
  __int64 v56; // rcx
  unsigned int v57; // r8d
  __int64 v58; // r15
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 *v61; // rax
  __int64 v62; // r15
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 *v65; // rax
  __int64 v66; // r15
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // r15
  __int64 v70; // rax
  __int64 v71; // rax
  char v72; // [rsp+0h] [rbp-90h]
  char v73; // [rsp+0h] [rbp-90h]
  __int64 v74; // [rsp+8h] [rbp-88h]
  __int64 v75; // [rsp+8h] [rbp-88h]
  unsigned int v76; // [rsp+8h] [rbp-88h]
  bool v77; // [rsp+8h] [rbp-88h]
  unsigned int v78; // [rsp+8h] [rbp-88h]
  unsigned int v79; // [rsp+8h] [rbp-88h]
  bool v80; // [rsp+8h] [rbp-88h]
  __int64 v81; // [rsp+8h] [rbp-88h]
  char v82; // [rsp+8h] [rbp-88h]
  bool v83; // [rsp+8h] [rbp-88h]
  char v84; // [rsp+8h] [rbp-88h]
  bool v85; // [rsp+8h] [rbp-88h]
  unsigned int v86; // [rsp+8h] [rbp-88h]
  char v87; // [rsp+8h] [rbp-88h]
  __int64 v88; // [rsp+8h] [rbp-88h]
  char v89; // [rsp+10h] [rbp-80h]
  int v90; // [rsp+1Ch] [rbp-74h]
  unsigned int v91; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 v92; // [rsp+30h] [rbp-60h] BYREF
  int v93; // [rsp+38h] [rbp-58h]
  __int64 v94; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v95; // [rsp+48h] [rbp-48h]
  __int64 v96[8]; // [rsp+50h] [rbp-40h] BYREF

  v90 = a5;
  if ( (unsigned int)a5 > 2 )
    return 0;
  while ( 1 )
  {
    v9 = *a3;
    if ( *(_WORD *)(*a3 + 24) )
    {
      v89 = 0;
    }
    else
    {
      v10 = *a4;
      if ( !*(_WORD *)(*a4 + 24) )
      {
        v11 = sub_15A35F0(*(unsigned __int16 *)a2, *(_QWORD *)(v9 + 32), *(_QWORD *)(v10 + 32), 0, a5, a6);
        if ( (unsigned __int8)sub_1593BB0(v11) )
          goto LABEL_21;
LABEL_5:
        v12 = sub_15E0530(*(_QWORD *)(a1 + 24));
        v13 = sub_159C540(v12);
        v14 = sub_145CE20(a1, v13);
        *a4 = v14;
        *a3 = v14;
        *a2 = 32;
        return 1;
      }
      *a3 = v10;
      *a4 = v9;
      v89 = 1;
      *a2 = sub_15FF5D0(*a2);
    }
    v16 = *a4;
    v17 = *(_WORD *)(*a4 + 24);
    if ( v17 == 7 )
    {
      v18 = *(_QWORD *)(v16 + 48);
      if ( sub_146CEE0(a1, *a3, v18) && (v40 = sub_146D930(a1, *a3, **(_QWORD **)(v18 + 32))) )
      {
        v41 = *a3;
        *a3 = *a4;
        *a4 = v41;
        v89 = v40;
        *a2 = sub_15FF5D0(*a2);
        v16 = *a4;
        v17 = *(_WORD *)(*a4 + 24);
      }
      else
      {
        v16 = *a4;
        v17 = *(_WORD *)(*a4 + 24);
      }
    }
    if ( v17 )
      goto LABEL_28;
    v19 = *(_QWORD *)(v16 + 32);
    v20 = *a2;
    v74 = v19 + 24;
    if ( (unsigned int)(v20 - 32) > 1 )
      break;
LABEL_12:
    v21 = sub_13D01C0(v74);
    v22 = *a3;
    if ( !v21
      || *(_WORD *)(v22 + 24) != 4
      || (v23 = *(__int64 **)(v22 + 32), v24 = *v23, *(_WORD *)(*v23 + 24) != 5)
      || *(_QWORD *)(v22 + 40) != 2
      || *(_QWORD *)(v24 + 40) != 2 )
    {
      v16 = *a4;
      goto LABEL_29;
    }
    v75 = *v23;
    v25 = sub_1456170(**(_QWORD **)(v24 + 32));
    if ( !v25 )
    {
LABEL_27:
      v16 = *a4;
LABEL_28:
      v22 = *a3;
      goto LABEL_29;
    }
    v89 = v25;
    *a4 = *(_QWORD *)(*(_QWORD *)(v22 + 32) + 8LL);
    v22 = *(_QWORD *)(*(_QWORD *)(v75 + 32) + 8LL);
    *a3 = v22;
    v16 = *a4;
LABEL_29:
    if ( (unsigned __int8)sub_1452FA0(v22, v16) )
    {
      if ( (unsigned __int8)sub_15FF820(*a2) )
        goto LABEL_5;
      if ( (unsigned __int8)sub_15FF850(*a2) )
        goto LABEL_21;
    }
    v30 = *a2;
    if ( *a2 == 39 )
    {
      v54 = sub_1477920(a1, *a4, 1u);
      sub_158ACE0(&v94, v54);
      v84 = sub_13CFF40(&v94, (__int64)v54, v55, v56, v57);
      sub_135E100(&v94);
      if ( v84 )
      {
        v65 = sub_1477920(a1, *a3, 1u);
        sub_158ABC0(&v94, v65);
        if ( v95 <= 0x40 )
        {
          v73 = v95;
          v88 = v94;
          sub_135E100(&v94);
          if ( v88 == (1LL << (v73 - 1)) - 1 )
            goto LABEL_47;
        }
        else
        {
          if ( (*(_QWORD *)(v94 + 8LL * ((v95 - 1) >> 6)) & (1LL << ((unsigned __int8)v95 - 1))) == 0 )
          {
            v86 = v95 - 1;
            if ( v86 == (unsigned int)sub_16A58F0(&v94) )
            {
              sub_135E100(&v94);
LABEL_47:
              if ( !v89 )
                return 0;
              goto LABEL_48;
            }
          }
          sub_135E100(&v94);
        }
        v66 = *a3;
        v67 = sub_1456040(*a4);
        v68 = sub_145CF80(a1, v67, 1, 1u);
        *a3 = sub_13A5B00(a1, v68, v66, 4u, 0);
        *a2 = 38;
      }
      else
      {
        v58 = *a4;
        v59 = sub_1456040(*a4);
        v60 = sub_145CF80(a1, v59, -1, 1u);
        *a4 = sub_13A5B00(a1, v60, v58, 4u, 0);
        *a2 = 38;
      }
    }
    else if ( v30 > 0x27 )
    {
      if ( v30 != 41 )
        goto LABEL_47;
      v35 = sub_1477920(a1, *a4, 1u);
      sub_158ABC0(&v94, v35);
      if ( v95 <= 0x40 )
      {
        v72 = v95;
        v81 = v94;
        sub_135E100(&v94);
        if ( v81 != (1LL << (v72 - 1)) - 1 )
          goto LABEL_43;
      }
      else
      {
        if ( (*(_QWORD *)(v94 + 8LL * ((v95 - 1) >> 6)) & (1LL << ((unsigned __int8)v95 - 1))) != 0
          || (v78 = v95 - 1, v78 != (unsigned int)sub_16A58F0(&v94)) )
        {
          sub_135E100(&v94);
LABEL_43:
          v36 = *a4;
          v37 = sub_1456040(*a4);
          v38 = sub_145CF80(a1, v37, 1, 1u);
          *a4 = sub_13A5B00(a1, v38, v36, 4u, 0);
          *a2 = 40;
          goto LABEL_48;
        }
        sub_135E100(&v94);
      }
      v42 = sub_1477920(a1, *a3, 1u);
      sub_158ACE0(&v94, v42);
      v82 = sub_13CFF40(&v94, (__int64)v42, v43, v44, v45);
      sub_135E100(&v94);
      if ( v82 )
        goto LABEL_47;
      v46 = *a3;
      v47 = sub_1456040(*a4);
      v48 = sub_145CF80(a1, v47, -1, 1u);
      *a3 = sub_13A5B00(a1, v48, v46, 4u, 0);
      *a2 = 40;
    }
    else if ( v30 == 35 )
    {
      v50 = sub_1477920(a1, *a4, 0);
      sub_158AAD0(&v94, v50);
      v83 = sub_13D01C0((__int64)&v94);
      sub_135E100(&v94);
      if ( v83 )
      {
        v39 = sub_1477920(a1, *a3, 0);
        sub_158A9F0(&v94, v39);
        if ( v95 <= 0x40 )
        {
          v80 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v95) == v94;
        }
        else
        {
          v79 = v95;
          v80 = v79 == (unsigned int)sub_16A58F0(&v94);
        }
        sub_135E100(&v94);
        if ( v80 )
          goto LABEL_47;
        v69 = *a3;
        v70 = sub_1456040(*a4);
        v71 = sub_145CF80(a1, v70, 1, 1u);
        *a3 = sub_13A5B00(a1, v71, v69, 2u, 0);
        *a2 = 34;
      }
      else
      {
        v51 = *a4;
        v52 = sub_1456040(*a4);
        v53 = sub_145CF80(a1, v52, -1, 1u);
        *a4 = sub_13A5B00(a1, v53, v51, 0, 0);
        *a2 = 34;
      }
    }
    else
    {
      if ( v30 != 37 )
        goto LABEL_47;
      v31 = sub_1477920(a1, *a4, 0);
      sub_158A9F0(&v94, v31);
      if ( v95 <= 0x40 )
      {
        v77 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v95) == v94;
      }
      else
      {
        v76 = v95;
        v77 = v76 == (unsigned int)sub_16A58F0(&v94);
      }
      sub_135E100(&v94);
      if ( v77 )
      {
        v61 = sub_1477920(a1, *a3, 0);
        sub_158AAD0(&v94, v61);
        v85 = sub_13D01C0((__int64)&v94);
        sub_135E100(&v94);
        if ( v85 )
          goto LABEL_47;
        v62 = *a3;
        v63 = sub_1456040(*a4);
        v64 = sub_145CF80(a1, v63, -1, 1u);
        *a3 = sub_13A5B00(a1, v64, v62, 0, 0);
        *a2 = 36;
      }
      else
      {
        v32 = *a4;
        v33 = sub_1456040(*a4);
        v34 = sub_145CF80(a1, v33, 1, 1u);
        *a4 = sub_13A5B00(a1, v34, v32, 2u, 0);
        *a2 = 36;
      }
    }
LABEL_48:
    if ( ++v90 == 3 )
      return 0;
  }
  sub_158B890(&v94, v20, v19 + 24);
  if ( (unsigned __int8)sub_158A0B0(&v94) )
  {
    sub_135E100(v96);
    sub_135E100(&v94);
    goto LABEL_5;
  }
  if ( !(unsigned __int8)sub_158A120(&v94) )
  {
    v93 = 1;
    v92 = 0;
    v29 = sub_158A180(&v94, &v91, &v92);
    if ( v29 && v91 - 32 <= 1 )
    {
      *a2 = v91;
      v87 = v29;
      *a4 = sub_145CF40(a1, (__int64)&v92);
      sub_135E100(&v92);
      sub_135E100(v96);
      sub_135E100(&v94);
      v16 = *a4;
      v22 = *a3;
      v89 = v87;
    }
    else
    {
      sub_135E100(&v92);
      sub_135E100(v96);
      sub_135E100(&v94);
      switch ( *a2 )
      {
        case ' ':
        case '!':
          goto LABEL_12;
        case '#':
          *a2 = 34;
          goto LABEL_62;
        case '%':
          *a2 = 36;
          goto LABEL_59;
        case '\'':
          *a2 = 38;
LABEL_62:
          sub_13A38D0((__int64)&v92, v74);
          sub_16A7800(&v92, 1);
          break;
        case ')':
          *a2 = 40;
LABEL_59:
          sub_13A38D0((__int64)&v92, v74);
          sub_16A7490(&v92, 1);
          break;
        default:
          goto LABEL_27;
      }
      v49 = v93;
      v93 = 0;
      v95 = v49;
      v94 = v92;
      *a4 = sub_145CF40(a1, (__int64)&v94);
      sub_135E100(&v94);
      sub_135E100(&v92);
      v16 = *a4;
      v22 = *a3;
      v89 = 1;
    }
    goto LABEL_29;
  }
  sub_135E100(v96);
  sub_135E100(&v94);
LABEL_21:
  v26 = sub_15E0530(*(_QWORD *)(a1 + 24));
  v27 = sub_159C540(v26);
  v28 = sub_145CE20(a1, v27);
  *a4 = v28;
  *a3 = v28;
  *a2 = 33;
  return 1;
}
