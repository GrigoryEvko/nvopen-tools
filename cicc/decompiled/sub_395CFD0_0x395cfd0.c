// Function: sub_395CFD0
// Address: 0x395cfd0
//
__int64 __fastcall sub_395CFD0(unsigned __int16 *a1, __int64 *a2, __int64 a3, _QWORD *a4)
{
  __int64 v5; // r12
  __int64 *v6; // rbx
  unsigned __int16 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  int v13; // r9d
  __int64 v14; // r15
  __int64 v15; // rax
  _BYTE *v17; // rcx
  int v18; // ebx
  int v19; // edx
  __int64 v20; // r8
  __int64 v21; // rax
  unsigned int v22; // r15d
  _QWORD *v23; // rax
  unsigned __int8 *v24; // rsi
  _QWORD *v25; // rdi
  __int64 **v26; // r11
  _QWORD *v27; // rdx
  unsigned __int64 v28; // rcx
  _BYTE *v29; // r15
  unsigned __int64 v30; // r14
  __int64 v31; // rax
  _QWORD *v32; // rdx
  __int64 v33; // rsi
  unsigned __int64 v34; // rcx
  _QWORD *v35; // rax
  int v36; // eax
  __int64 v37; // rax
  int v38; // r8d
  int v39; // r9d
  __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 *v43; // rdx
  unsigned __int64 v44; // r8
  __int64 *v45; // r13
  __int64 *v46; // r12
  unsigned __int64 v47; // r14
  __int64 v48; // r9
  __int64 *v49; // rbx
  __int64 *v50; // rax
  __int64 v51; // rcx
  _QWORD *v52; // rax
  __int64 *v53; // rbx
  __int64 v54; // r12
  _QWORD *v55; // r13
  __int64 v56; // rax
  _QWORD *v57; // rsi
  __int64 v58; // rsi
  _QWORD *v59; // r10
  _BYTE *v60; // r13
  __int64 v61; // r9
  __int64 *v62; // r12
  bool v63; // si
  __int64 v64; // rbx
  unsigned __int64 v65; // r14
  _QWORD *v66; // r8
  _QWORD *v67; // rax
  __int64 v68; // rax
  char v69; // di
  _QWORD *v70; // [rsp+8h] [rbp-1A8h]
  __int64 v71; // [rsp+10h] [rbp-1A0h]
  __int64 v72; // [rsp+20h] [rbp-190h]
  _QWORD *v73; // [rsp+20h] [rbp-190h]
  unsigned __int64 *v74; // [rsp+28h] [rbp-188h]
  __int64 *v75; // [rsp+28h] [rbp-188h]
  _QWORD *v76; // [rsp+30h] [rbp-180h]
  __int64 v77; // [rsp+30h] [rbp-180h]
  unsigned __int64 v78; // [rsp+38h] [rbp-178h]
  __int64 v79; // [rsp+38h] [rbp-178h]
  int v80; // [rsp+58h] [rbp-158h]
  unsigned __int64 v81; // [rsp+60h] [rbp-150h] BYREF
  unsigned int v82; // [rsp+68h] [rbp-148h]
  char v83; // [rsp+70h] [rbp-140h]
  char v84; // [rsp+71h] [rbp-13Fh]
  _BYTE *v85; // [rsp+80h] [rbp-130h] BYREF
  __int64 v86; // [rsp+88h] [rbp-128h]
  _BYTE v87[32]; // [rsp+90h] [rbp-120h] BYREF
  unsigned __int8 *v88; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v89; // [rsp+B8h] [rbp-F8h]
  _BYTE v90[32]; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v91[3]; // [rsp+E0h] [rbp-D0h] BYREF
  _QWORD *v92; // [rsp+F8h] [rbp-B8h]
  __int64 v93; // [rsp+100h] [rbp-B0h]
  int v94; // [rsp+108h] [rbp-A8h]
  __int64 v95; // [rsp+110h] [rbp-A0h]
  __int64 v96; // [rsp+118h] [rbp-98h]
  unsigned __int64 v97[2]; // [rsp+130h] [rbp-80h] BYREF
  __int64 *v98; // [rsp+140h] [rbp-70h]
  __int64 v99; // [rsp+148h] [rbp-68h]
  __int64 v100; // [rsp+150h] [rbp-60h]
  unsigned __int64 v101; // [rsp+158h] [rbp-58h]
  __int64 v102; // [rsp+160h] [rbp-50h]
  __int64 v103; // [rsp+168h] [rbp-48h]
  __int64 v104; // [rsp+170h] [rbp-40h]
  unsigned __int64 v105; // [rsp+178h] [rbp-38h]

  v5 = a3;
  v6 = a2;
  v7 = sub_14AC610(a1, a2, a3);
  v8 = sub_1649C60((__int64)v7);
  v9 = v8;
  if ( *(_BYTE *)(v8 + 16) == 56 && !(unsigned __int8)sub_15FA290(v8) )
  {
    v80 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
    v97[1] = 8;
    v97[0] = sub_22077B0(0x40u);
    v11 = v97[0] + 24;
    v12 = sub_22077B0(0x200u);
    v101 = v97[0] + 24;
    *(_QWORD *)(v97[0] + 24) = v12;
    v99 = v12;
    v100 = v12 + 512;
    v105 = v11;
    v103 = v12;
    v104 = v12 + 512;
    v98 = (__int64 *)v12;
    v102 = v12;
    if ( v80 )
    {
      v14 = (unsigned int)(v80 - 1);
      while ( 1 )
      {
        v15 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
        if ( *(_BYTE *)(*(_QWORD *)(v9 + 24 * (v14 - v15)) + 16LL) != 13 )
          break;
        v91[0] = *(_QWORD *)(v9 + 24 * (v14 - v15));
        sub_395CDA0((__int64)v97, v91);
        if ( v14-- == 0 )
        {
          sub_395CD20(v97);
          return v9;
        }
      }
      v85 = v87;
      v86 = 0x400000000LL;
      if ( (_DWORD)v14 )
      {
        v17 = v87;
        v18 = v14;
        v19 = 1;
        v20 = *(_QWORD *)(v9 + 24 * (1 - v15));
        v21 = 0;
        while ( 1 )
        {
          *(_QWORD *)&v17[8 * v21] = v20;
          v22 = v19 + 1;
          v21 = (unsigned int)(v86 + 1);
          LODWORD(v86) = v86 + 1;
          if ( v19 == v18 )
            break;
          v20 = *(_QWORD *)(v9 + 24 * (v22 - (unsigned __int64)(*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
          if ( HIDWORD(v86) <= (unsigned int)v21 )
          {
            v77 = *(_QWORD *)(v9 + 24 * (v22 - (unsigned __int64)(*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
            sub_16CD150((__int64)&v85, v87, 0, 8, v20, v13);
            v21 = (unsigned int)v86;
            v20 = v77;
          }
          v17 = v85;
          v19 = v22;
        }
        v6 = a2;
      }
      v23 = (_QWORD *)sub_16498A0(v9);
      v91[0] = 0;
      v92 = v23;
      v93 = 0;
      v94 = 0;
      v95 = 0;
      v96 = 0;
      v91[1] = *(_QWORD *)(v9 + 40);
      v91[2] = v9 + 24;
      v24 = *(unsigned __int8 **)(v9 + 48);
      v88 = v24;
      if ( v24 )
      {
        sub_1623A60((__int64)&v88, (__int64)v24, 2);
        if ( v91[0] )
          sub_161E7C0((__int64)v91, v91[0]);
        v91[0] = (__int64)v88;
        if ( v88 )
          sub_1623210((__int64)&v88, v88, (__int64)v91);
      }
      v25 = (_QWORD *)a4[2];
      v26 = (__int64 **)v85;
      v27 = a4 + 1;
      v76 = a4 + 1;
      v74 = (unsigned __int64 *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      v28 = *v74;
      v78 = (unsigned int)v86;
      while ( v25 )
      {
        while ( v25[4] < v28 )
        {
          v25 = (_QWORD *)v25[3];
          if ( !v25 )
            goto LABEL_25;
        }
        v52 = (_QWORD *)v25[2];
        if ( v25[4] <= v28 )
        {
          v57 = (_QWORD *)v25[3];
          while ( v57 )
          {
            if ( v57[4] <= v28 )
            {
              v57 = (_QWORD *)v57[3];
            }
            else
            {
              v27 = v57;
              v57 = (_QWORD *)v57[2];
            }
          }
          while ( v52 )
          {
            while ( 1 )
            {
              v58 = v52[3];
              if ( v52[4] >= v28 )
                break;
              v52 = (_QWORD *)v52[3];
              if ( !v58 )
                goto LABEL_70;
            }
            v25 = v52;
            v52 = (_QWORD *)v52[2];
          }
LABEL_70:
          if ( v25 != v27 )
          {
            v59 = a4;
            v60 = &v85[8 * (unsigned int)v86];
            v61 = v5;
            v62 = v6;
            v63 = v60 == v85;
            v64 = v9;
            v65 = (unsigned __int64)v85;
            do
            {
              v29 = (_BYTE *)v25[5];
              v66 = (_QWORD *)v65;
              v67 = &v29[24 * (1LL - (*((_DWORD *)v29 + 5) & 0xFFFFFFF))];
              if ( v29 != (_BYTE *)v67 && !v63 )
              {
                do
                {
                  if ( *v67 != *v66 )
                    break;
                  v67 += 3;
                  ++v66;
                  if ( v29 == (_BYTE *)v67 )
                    goto LABEL_79;
                }
                while ( v60 != (_BYTE *)v66 );
              }
              if ( v29 == (_BYTE *)v67 )
              {
LABEL_79:
                if ( v60 == (_BYTE *)v66 )
                {
                  v6 = v62;
                  v5 = v61;
                  goto LABEL_34;
                }
              }
              v70 = v59;
              v71 = v61;
              v73 = v27;
              v68 = sub_220EEE0((__int64)v25);
              v27 = v73;
              v61 = v71;
              v59 = v70;
              v25 = (_QWORD *)v68;
            }
            while ( (_QWORD *)v68 != v73 );
            v26 = (__int64 **)v65;
            a4 = v70;
            v9 = v64;
            v6 = v62;
            v5 = v71;
          }
          break;
        }
        v27 = v25;
        v25 = (_QWORD *)v25[2];
      }
LABEL_25:
      v90[1] = 1;
      v88 = "splitGEPI.base";
      v90[0] = 3;
      v29 = (_BYTE *)sub_1BBF860(v91, *(_QWORD *)(v9 + 56), (_BYTE *)*v74, v26, v78, (__int64 *)&v88);
      v30 = *(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      v31 = sub_22077B0(0x30u);
      v32 = (_QWORD *)a4[2];
      *(_QWORD *)(v31 + 32) = v30;
      v33 = v31;
      *(_QWORD *)(v31 + 40) = v29;
      if ( v32 )
      {
        while ( 1 )
        {
          v34 = v32[4];
          v35 = (_QWORD *)v32[3];
          if ( v30 < v34 )
            v35 = (_QWORD *)v32[2];
          if ( !v35 )
            break;
          v32 = v35;
        }
        v69 = 1;
        if ( v76 != v32 )
          v69 = v30 < v34;
      }
      else
      {
        v32 = v76;
        v69 = 1;
      }
      sub_220F040(v69, v33, v32, v76);
      ++a4[5];
LABEL_34:
      v79 = *(_QWORD *)v29;
      v36 = sub_15A9520(v5, *(_DWORD *)(*(_QWORD *)v29 + 8LL) >> 8);
      v88 = v90;
      v89 = 0x400000000LL;
      v37 = sub_1644900(v92, 8 * v36);
      v40 = sub_159C470(v37, 0, 0);
      v41 = (unsigned int)v89;
      if ( (unsigned int)v89 >= HIDWORD(v89) )
      {
        sub_16CD150((__int64)&v88, v90, 0, 8, v38, v39);
        v41 = (unsigned int)v89;
      }
      v42 = v5;
      *(_QWORD *)&v88[8 * v41] = v40;
      v43 = v6;
      v44 = (unsigned int)(v89 + 1);
      v45 = v98;
      LODWORD(v89) = v89 + 1;
      v46 = (__int64 *)v102;
      v47 = v101;
      v48 = v42;
      v49 = (__int64 *)v100;
      v50 = v43;
      while ( v46 != v45 )
      {
        if ( HIDWORD(v89) <= (unsigned int)v44 )
        {
          v72 = v48;
          v75 = v50;
          sub_16CD150((__int64)&v88, v90, 0, 8, v44, v48);
          v44 = (unsigned int)v89;
          v48 = v72;
          v50 = v75;
        }
        v51 = *v45++;
        *(_QWORD *)&v88[8 * v44] = v51;
        v44 = (unsigned int)(v89 + 1);
        LODWORD(v89) = v89 + 1;
        if ( v49 == v45 )
        {
          v45 = *(__int64 **)(v47 + 8);
          v47 += 8LL;
          v49 = v45 + 64;
        }
      }
      v53 = v50;
      v54 = v48;
      v84 = 1;
      v81 = (unsigned __int64)"splitGEPI.replace";
      v83 = 3;
      v55 = (_QWORD *)sub_1BBF860(v91, *((_QWORD *)v29 + 8), v29, (__int64 **)v88, v44, (__int64 *)&v81);
      v82 = 8 * sub_15A9520(v54, *(_DWORD *)(v79 + 8) >> 8);
      if ( v82 > 0x40 )
        sub_16A4EF0((__int64)&v81, 0, 1);
      else
        v81 = 0;
      sub_15FA310((__int64)v55, v54, (__int64)&v81);
      if ( v82 > 0x40 )
        v56 = *(_QWORD *)v81;
      else
        v56 = (__int64)(v81 << (64 - (unsigned __int8)v82)) >> (64 - (unsigned __int8)v82);
      *v53 += v56;
      sub_15F20C0(v55);
      if ( v82 > 0x40 && v81 )
        j_j___libc_free_0_0(v81);
      if ( v88 != v90 )
        _libc_free((unsigned __int64)v88);
      if ( v91[0] )
        sub_161E7C0((__int64)v91, v91[0]);
      if ( v85 != v87 )
        _libc_free((unsigned __int64)v85);
      v9 = (__int64)v29;
      sub_395CD20(v97);
    }
    else
    {
      sub_395CD20(v97);
    }
  }
  return v9;
}
