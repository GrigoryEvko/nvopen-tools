// Function: sub_33CD9D0
// Address: 0x33cd9d0
//
__int64 __fastcall sub_33CD9D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned int v11; // edx
  unsigned __int16 v12; // cx
  __int64 v13; // rax
  int v14; // ebx
  int v15; // eax
  unsigned int v16; // r9d
  bool v18; // r9
  bool v19; // al
  _QWORD *v20; // r12
  __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // rax
  bool v24; // al
  unsigned int v25; // ebx
  unsigned int v26; // eax
  _QWORD *v27; // rax
  __int64 v28; // r12
  __int64 v29; // rbx
  bool v30; // zf
  unsigned __int64 v31; // rax
  unsigned __int8 v32; // al
  unsigned int v33; // edx
  unsigned __int8 v34; // r9
  unsigned __int64 v35; // rax
  _QWORD *v36; // rax
  __int64 v37; // rsi
  __int64 v38; // rbx
  __int64 v39; // rax
  __int16 v40; // dx
  __int64 v41; // rax
  bool v42; // al
  unsigned int v43; // eax
  _QWORD *v44; // rax
  __int64 v45; // rax
  unsigned __int16 v46; // dx
  __int64 v47; // rax
  unsigned int v48; // ebx
  unsigned int v49; // r13d
  bool v50; // al
  bool v51; // al
  bool v52; // al
  unsigned int v53; // ebx
  unsigned int v54; // ebx
  __int64 v55; // rbx
  _DWORD *v56; // r13
  __int64 v57; // r12
  __int64 v58; // rax
  bool v59; // r9
  __int64 v60; // r13
  __int64 v61; // r15
  signed int v62; // r12d
  bool v63; // bl
  bool v64; // al
  unsigned __int8 v65; // r9
  bool v66; // r8
  _QWORD *v67; // rax
  __int64 v68; // [rsp+8h] [rbp-C8h]
  __int64 *v69; // [rsp+8h] [rbp-C8h]
  __int64 v70; // [rsp+10h] [rbp-C0h]
  __int64 v71; // [rsp+10h] [rbp-C0h]
  unsigned int v72; // [rsp+18h] [rbp-B8h]
  unsigned int v73; // [rsp+18h] [rbp-B8h]
  __int64 *v74; // [rsp+18h] [rbp-B8h]
  unsigned int v75; // [rsp+20h] [rbp-B0h]
  bool v76; // [rsp+20h] [rbp-B0h]
  __int64 v77; // [rsp+20h] [rbp-B0h]
  bool v78; // [rsp+20h] [rbp-B0h]
  int v79; // [rsp+20h] [rbp-B0h]
  bool v80; // [rsp+20h] [rbp-B0h]
  unsigned __int16 v81; // [rsp+28h] [rbp-A8h]
  bool v82; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v83; // [rsp+28h] [rbp-A8h]
  __int64 v84; // [rsp+28h] [rbp-A8h]
  unsigned __int8 v85; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v86; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v87; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v88; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v89; // [rsp+30h] [rbp-A0h]
  __int64 v90; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v91; // [rsp+30h] [rbp-A0h]
  unsigned int v92; // [rsp+38h] [rbp-98h]
  unsigned int v93; // [rsp+3Ch] [rbp-94h] BYREF
  unsigned __int16 v94; // [rsp+40h] [rbp-90h] BYREF
  __int64 v95; // [rsp+48h] [rbp-88h]
  unsigned __int16 v96; // [rsp+50h] [rbp-80h] BYREF
  __int64 v97; // [rsp+58h] [rbp-78h]
  unsigned __int64 v98; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v99; // [rsp+68h] [rbp-68h]
  unsigned __int64 v100; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v101; // [rsp+78h] [rbp-58h]
  unsigned __int64 v102; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v103; // [rsp+88h] [rbp-48h]
  unsigned int *v104; // [rsp+90h] [rbp-40h] BYREF
  __int64 v105; // [rsp+98h] [rbp-38h]

  v9 = a2;
  v10 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v93 = a6;
  v11 = *(_DWORD *)(a4 + 8);
  v12 = *(_WORD *)v10;
  v13 = *(_QWORD *)(v10 + 8);
  v14 = *(_DWORD *)(a2 + 24);
  v94 = v12;
  v95 = v13;
  if ( v11 <= 0x40 )
  {
    if ( !*(_QWORD *)a4 )
      return 0;
  }
  else
  {
    v75 = v11;
    v81 = v12;
    v15 = sub_C444A0(a4);
    v11 = v75;
    v12 = v81;
    if ( v75 == v15 )
      return 0;
  }
  if ( v93 > 5 )
    return 0;
  if ( v14 > 57 )
  {
    switch ( v14 )
    {
      case 168:
        v30 = *(_DWORD *)(**(_QWORD **)(a2 + 40) + 24LL) == 51;
        LODWORD(v105) = v11;
        if ( v30 )
        {
          if ( v11 > 0x40 )
          {
            sub_C43690((__int64)&v104, -1, 1);
          }
          else
          {
            v31 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
            if ( !v11 )
              v31 = 0;
            v104 = (unsigned int *)v31;
          }
        }
        else if ( v11 > 0x40 )
        {
          sub_C43690((__int64)&v104, 0, 0);
        }
        else
        {
          v104 = 0;
        }
        if ( *(_DWORD *)(a5 + 8) > 0x40u && *(_QWORD *)a5 )
          j_j___libc_free_0_0(*(_QWORD *)a5);
        v16 = 1;
        *(_QWORD *)a5 = v104;
        *(_DWORD *)(a5 + 8) = v105;
        break;
      case 186:
      case 187:
      case 188:
        goto LABEL_22;
      case 189:
      case 213:
      case 214:
      case 216:
        v16 = sub_33CD9D0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a4, a5, v93 + 1);
        break;
      default:
        goto LABEL_10;
    }
    return v16;
  }
  if ( v14 > 55 )
  {
LABEL_22:
    v27 = *(_QWORD **)(a2 + 40);
    v99 = 1;
    v98 = 0;
    v101 = 1;
    v100 = 0;
    v28 = v27[5];
    v29 = v27[6];
    if ( (unsigned __int8)sub_33CD9D0(a1, *v27, v27[1], a4, &v98, v93 + 1)
      && (v32 = sub_33CD9D0(a1, v28, v29, a4, &v100, v93 + 1)) != 0 )
    {
      v88 = v32;
      sub_9865C0((__int64)&v102, (__int64)&v98);
      v33 = v103;
      v34 = v88;
      if ( v103 > 0x40 )
      {
        sub_C43BD0(&v102, (__int64 *)&v100);
        v33 = v103;
        v35 = v102;
        v34 = v88;
      }
      else
      {
        v35 = v100 | v102;
        v102 |= v100;
      }
      v89 = v34;
      LODWORD(v105) = v33;
      v104 = (unsigned int *)v35;
      v103 = 0;
      sub_33C8540(a5, (__int64)&v104);
      sub_969240((__int64 *)&v104);
      sub_969240((__int64 *)&v102);
      v16 = v89;
    }
    else
    {
      v16 = 0;
    }
    if ( v101 > 0x40 && v100 )
    {
      v86 = v16;
      j_j___libc_free_0_0(v100);
      v16 = v86;
    }
    if ( v99 > 0x40 && v98 )
    {
      v87 = v16;
      j_j___libc_free_0_0(v98);
      return v87;
    }
    return v16;
  }
LABEL_10:
  v18 = (unsigned int)v14 > 0x1F3 || (unsigned int)(v14 - 46) <= 2;
  if ( v18 )
    return (*(unsigned int (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 16) + 2112LL))(
             *(_QWORD *)(a1 + 16),
             a2,
             a3,
             a4,
             a5);
  if ( !v12 )
  {
    v19 = sub_3007100((__int64)&v94);
    v18 = 0;
    if ( !v19 )
      goto LABEL_15;
    return 0;
  }
  if ( (unsigned __int16)(v12 - 176) <= 0x34u )
    return 0;
LABEL_15:
  v76 = v18;
  v72 = sub_3281500(&v94, v93);
  sub_9691E0((__int64)&v104, v72, 0, 0, 0);
  sub_33C8540(a5, (__int64)&v104);
  sub_969240((__int64 *)&v104);
  v16 = v76;
  if ( v14 == 165 )
  {
    sub_9691E0((__int64)&v100, v72, 0, 0, 0);
    sub_9691E0((__int64)&v102, v72, 0, 0, 0);
    v58 = sub_3288400(a2, v72);
    v59 = v76;
    if ( v72 )
    {
      v69 = (__int64 *)a5;
      v60 = 0;
      v61 = v58;
      while ( 1 )
      {
        v62 = *(_DWORD *)(v61 + 4 * v60);
        if ( v62 < 0 )
        {
          sub_987080(v69, v60);
        }
        else if ( sub_986C60((__int64 *)a4, v60) )
        {
          if ( v62 < (int)v72 )
            sub_987080((__int64 *)&v100, v62);
          else
            sub_987080((__int64 *)&v102, v62 - v72);
        }
        if ( v72 - 1 == v60 )
          break;
        ++v60;
      }
      v59 = v76;
      v9 = a2;
    }
    v80 = v59;
    v63 = sub_9867B0((__int64)&v100);
    v64 = sub_9867B0((__int64)&v102);
    v65 = v80;
    if ( v63 )
    {
      if ( !v64 )
      {
LABEL_88:
        v104 = &v93;
        v105 = a1;
        v66 = sub_9867B0((__int64)&v100);
        v67 = *(_QWORD **)(v9 + 40);
        if ( v66 )
          v65 = sub_33CE520(&v104, v67[5], v67[6], &v102);
        else
          v65 = sub_33CE520(&v104, *v67, v67[1], &v100);
      }
    }
    else
    {
      v65 = 0;
      if ( v64 )
        goto LABEL_88;
    }
    v91 = v65;
    sub_969240((__int64 *)&v102);
    sub_969240((__int64 *)&v100);
    return v91;
  }
  if ( v14 > 165 )
  {
    if ( v14 <= 225 )
    {
      if ( v14 <= 222 )
        return v16;
      v36 = *(_QWORD **)(a2 + 40);
      v37 = v36[1];
      v38 = *v36;
      v39 = *(_QWORD *)(*v36 + 48LL) + 16LL * *((unsigned int *)v36 + 2);
      v40 = *(_WORD *)v39;
      v41 = *(_QWORD *)(v39 + 8);
      LOWORD(v104) = v40;
      v105 = v41;
      v42 = sub_3280200((__int64)&v104);
      v16 = v76;
      if ( v42 )
        return v16;
      v43 = sub_3281500(&v104, v37);
      v101 = 1;
      v100 = 0;
      sub_C449B0((__int64)&v102, (const void **)a4, v43);
      v85 = sub_33CD9D0(a1, v38, v37, &v102, &v100, v93 + 1);
      if ( !v85 )
      {
LABEL_21:
        sub_969240((__int64 *)&v102);
        sub_969240((__int64 *)&v100);
        return v85;
      }
      sub_C44740((__int64)&v104, (char **)&v100, v72);
LABEL_53:
      sub_33C8540(a5, (__int64)&v104);
      sub_969240((__int64 *)&v104);
      goto LABEL_21;
    }
    if ( v14 == 234 )
    {
      v44 = *(_QWORD **)(a2 + 40);
      v83 = v44[1];
      v71 = *v44;
      v68 = *((unsigned int *)v44 + 2);
      v45 = *(_QWORD *)(*v44 + 48LL) + 16 * v68;
      v46 = *(_WORD *)v45;
      v47 = *(_QWORD *)(v45 + 8);
      v96 = v46;
      v97 = v47;
      v48 = sub_32844A0(&v96, (__int64)&v104);
      v49 = sub_32844A0(&v94, (__int64)&v104);
      v50 = sub_32801E0((__int64)&v96);
      v16 = v76;
      if ( v50 )
      {
        v51 = sub_3280180((__int64)&v96);
        v16 = v76;
        if ( v51 )
        {
          v52 = sub_3280180((__int64)&v94);
          v16 = v76;
          v78 = v52;
          if ( v52 )
          {
            v73 = v49 / v48;
            if ( !(v49 % v48) )
            {
              v53 = sub_3281500(&v96, (__int64)&v104);
              sub_C4DEC0((__int64)&v98, a4, v53, 0);
              v92 = v53;
              v54 = 0;
              while ( v54 != v73 )
              {
                v101 = 1;
                v100 = 0;
                sub_9866F0((__int64)&v102, v73, v54);
                sub_C47700((__int64)&v104, v92, (__int64)&v102);
                sub_33C8520(&v104, (__int64 *)&v98);
                v83 = v68 | v83 & 0xFFFFFFFF00000000LL;
                if ( !(unsigned __int8)sub_33CD9D0(a1, v71, v83, &v104, &v100, v93 + 1) || !sub_9867B0((__int64)&v100) )
                {
                  sub_969240((__int64 *)&v104);
                  sub_969240((__int64 *)&v102);
                  sub_969240((__int64 *)&v100);
                  v78 = 0;
                  break;
                }
                ++v54;
                sub_969240((__int64 *)&v104);
                sub_969240((__int64 *)&v102);
                sub_969240((__int64 *)&v100);
              }
              sub_969240((__int64 *)&v98);
              return v78;
            }
          }
        }
      }
    }
  }
  else
  {
    if ( v14 != 156 )
    {
      if ( v14 != 161 )
        return v16;
      v20 = *(_QWORD **)(a2 + 40);
      v82 = v76;
      v77 = *v20;
      v70 = v20[1];
      v21 = *(_QWORD *)(*v20 + 48LL) + 16LL * *((unsigned int *)v20 + 2);
      v22 = *(_WORD *)v21;
      v23 = *(_QWORD *)(v21 + 8);
      LOWORD(v104) = v22;
      v105 = v23;
      v24 = sub_3280200((__int64)&v104);
      v16 = v82;
      if ( v24 )
        return v16;
      v25 = sub_33C84C0((__int64)v20, 1u);
      v26 = sub_3281500(&v104, 1);
      v101 = 1;
      v100 = 0;
      sub_C449B0((__int64)&v104, (const void **)a4, v26);
      sub_9865C0((__int64)&v102, (__int64)&v104);
      sub_33C90E0((__int64)&v102, v25);
      sub_969240((__int64 *)&v104);
      v85 = sub_33CD9D0(a1, v77, v70, &v102, &v100, v93 + 1);
      if ( !v85 )
        goto LABEL_21;
      sub_C440A0((__int64)&v104, (__int64 *)&v100, v72, v25);
      goto LABEL_53;
    }
    v79 = 0;
    v90 = 0;
    v84 = v72;
    v55 = 0;
    if ( v72 )
    {
      v74 = (__int64 *)a5;
      do
      {
        v56 = (_DWORD *)(*(_QWORD *)(a2 + 40) + 40 * v55);
        v57 = *(_QWORD *)v56;
        if ( *(_DWORD *)(*(_QWORD *)v56 + 24LL) == 51 )
        {
          sub_987080(v74, v55);
        }
        else if ( sub_986C60((__int64 *)a4, v55) )
        {
          if ( v90 && (v90 != v57 || v56[2] != v79) )
            return 0;
          v79 = v56[2];
          v90 = v57;
        }
        ++v55;
      }
      while ( v55 != v84 );
    }
    return 1;
  }
  return v16;
}
