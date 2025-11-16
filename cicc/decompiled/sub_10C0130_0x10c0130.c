// Function: sub_10C0130
// Address: 0x10c0130
//
__int64 __fastcall sub_10C0130(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r14
  __int64 v5; // rdi
  _BYTE *v7; // r13
  __int64 v8; // rdi
  int v9; // r15d
  __int64 v10; // r11
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rdx
  unsigned int v15; // eax
  unsigned int v16; // ebx
  _QWORD *v17; // rax
  __int64 *v18; // rbx
  __int64 v19; // r13
  __int64 v20; // r15
  unsigned int **v21; // r13
  _BYTE *v22; // rax
  __int64 v23; // rax
  char v24; // al
  _BYTE *v25; // rax
  __int64 v26; // rdx
  _BYTE *v27; // rax
  int v28; // eax
  __int64 v29; // rax
  unsigned int v30; // eax
  __int64 v31; // r13
  __int64 v32; // rbx
  __int64 v33; // rdx
  unsigned int v34; // esi
  bool v35; // al
  bool v36; // al
  unsigned int v37; // eax
  int v38; // eax
  unsigned int v39; // eax
  __int64 *v40; // rsi
  unsigned int **v41; // r13
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  unsigned int v45; // eax
  _BYTE *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdx
  char v49; // [rsp+20h] [rbp-190h]
  __int64 v50; // [rsp+28h] [rbp-188h]
  int v51; // [rsp+30h] [rbp-180h]
  __int64 v52; // [rsp+38h] [rbp-178h]
  int v53; // [rsp+48h] [rbp-168h]
  __int64 v54; // [rsp+48h] [rbp-168h]
  __int64 v57; // [rsp+58h] [rbp-158h]
  __int64 v58; // [rsp+58h] [rbp-158h]
  __int64 v59; // [rsp+58h] [rbp-158h]
  __int64 v60; // [rsp+58h] [rbp-158h]
  __int64 v61; // [rsp+58h] [rbp-158h]
  __int64 v62; // [rsp+58h] [rbp-158h]
  __int64 v63; // [rsp+58h] [rbp-158h]
  __int64 v64; // [rsp+58h] [rbp-158h]
  __int64 v65; // [rsp+60h] [rbp-150h] BYREF
  __int64 v66; // [rsp+68h] [rbp-148h] BYREF
  _QWORD **v67; // [rsp+70h] [rbp-140h] BYREF
  int v68; // [rsp+78h] [rbp-138h]
  _QWORD **v69; // [rsp+80h] [rbp-130h] BYREF
  int v70; // [rsp+88h] [rbp-128h]
  _QWORD *v71; // [rsp+90h] [rbp-120h] BYREF
  unsigned int v72; // [rsp+98h] [rbp-118h]
  _QWORD *v73; // [rsp+A0h] [rbp-110h] BYREF
  unsigned int v74; // [rsp+A8h] [rbp-108h]
  _QWORD **v75; // [rsp+B0h] [rbp-100h] BYREF
  unsigned int v76; // [rsp+B8h] [rbp-F8h]
  __int64 v77; // [rsp+C0h] [rbp-F0h] BYREF
  unsigned int v78; // [rsp+C8h] [rbp-E8h]
  _QWORD **v79; // [rsp+D0h] [rbp-E0h] BYREF
  unsigned int v80; // [rsp+D8h] [rbp-D8h]
  __int64 v81; // [rsp+E0h] [rbp-D0h] BYREF
  unsigned int v82; // [rsp+E8h] [rbp-C8h]
  __int64 v83; // [rsp+F0h] [rbp-C0h] BYREF
  unsigned int v84; // [rsp+F8h] [rbp-B8h]
  __int64 v85; // [rsp+100h] [rbp-B0h] BYREF
  unsigned int v86; // [rsp+108h] [rbp-A8h]
  char v87; // [rsp+110h] [rbp-A0h]
  _QWORD *v88; // [rsp+120h] [rbp-90h] BYREF
  unsigned int v89; // [rsp+128h] [rbp-88h]
  __int16 v90; // [rsp+140h] [rbp-70h]
  _QWORD **v91; // [rsp+150h] [rbp-60h] BYREF
  __int64 *v92; // [rsp+158h] [rbp-58h] BYREF
  __int64 v93; // [rsp+160h] [rbp-50h]
  unsigned int v94; // [rsp+168h] [rbp-48h]
  __int16 v95; // [rsp+170h] [rbp-40h]

  if ( !a2 )
    return 0;
  v4 = *(_QWORD *)(a2 - 64);
  if ( !v4 )
    return 0;
  v5 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v5 == 17 )
  {
    v52 = v5 + 24;
  }
  else
  {
    v13 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
    if ( (unsigned int)v13 > 1 )
      return 0;
    if ( *(_BYTE *)v5 > 0x15u )
      return 0;
    v25 = sub_AD7630(v5, 0, v13);
    if ( !v25 || *v25 != 17 )
      return 0;
    v52 = (__int64)(v25 + 24);
  }
  v53 = sub_B53900(a2);
  if ( !a3 )
    return 0;
  v7 = *(_BYTE **)(a3 - 64);
  if ( !v7 )
    return 0;
  v8 = *(_QWORD *)(a3 - 32);
  if ( *(_BYTE *)v8 != 17 )
  {
    v26 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
    if ( (unsigned int)v26 <= 1 && *(_BYTE *)v8 <= 0x15u )
    {
      v27 = sub_AD7630(v8, 0, v26);
      if ( v27 )
      {
        if ( *v27 == 17 )
        {
          v50 = (__int64)(v27 + 24);
          goto LABEL_9;
        }
      }
    }
    return 0;
  }
  v50 = v8 + 24;
LABEL_9:
  v65 = 0;
  v51 = sub_B53900(a3);
  v9 = v51;
  v66 = 0;
  if ( (_BYTE *)v4 == v7 )
    goto LABEL_13;
  LOBYTE(v93) = 0;
  v91 = &v88;
  v92 = &v65;
  if ( *(_BYTE *)v4 == 42 )
  {
    if ( *(_QWORD *)(v4 - 64) )
    {
      v88 = *(_QWORD **)(v4 - 64);
      if ( (unsigned __int8)sub_991580((__int64)&v92, *(_QWORD *)(v4 - 32)) )
        v4 = (__int64)v88;
    }
  }
  v91 = &v88;
  v92 = &v66;
  LOBYTE(v93) = 0;
  if ( *v7 == 42 )
  {
    if ( *((_QWORD *)v7 - 8) )
    {
      v88 = (_QWORD *)*((_QWORD *)v7 - 8);
      if ( (unsigned __int8)sub_991580((__int64)&v92, *((_QWORD *)v7 - 4)) )
        v7 = v88;
    }
  }
  v10 = 0;
  if ( v7 == (_BYTE *)v4 )
  {
LABEL_13:
    if ( a4 )
    {
      v28 = sub_B52870(v53);
      sub_AB1A50((__int64)&v75, v28, v52);
      v11 = v65;
      if ( !v65 )
        goto LABEL_28;
    }
    else
    {
      sub_AB1A50((__int64)&v75, v53, v52);
      v11 = v65;
      if ( !v65 )
        goto LABEL_29;
    }
    sub_AB1F90((__int64)&v91, (__int64 *)&v75, v11);
    if ( v76 > 0x40 && v75 )
      j_j___libc_free_0_0(v75);
    v75 = v91;
    v12 = (unsigned int)v92;
    LODWORD(v92) = 0;
    v76 = v12;
    if ( v78 > 0x40 && v77 )
    {
      j_j___libc_free_0_0(v77);
      v77 = v93;
      v78 = v94;
      if ( (unsigned int)v92 > 0x40 && v91 )
        j_j___libc_free_0_0(v91);
    }
    else
    {
      v77 = v93;
      v78 = v94;
    }
    if ( !a4 )
    {
LABEL_29:
      sub_AB1A50((__int64)&v79, v9, v50);
      if ( v66 )
      {
        sub_AB1F90((__int64)&v91, (__int64 *)&v79, v66);
        if ( v80 > 0x40 && v79 )
          j_j___libc_free_0_0(v79);
        v79 = v91;
        v15 = (unsigned int)v92;
        LODWORD(v92) = 0;
        v80 = v15;
        if ( v82 > 0x40 && v81 )
        {
          j_j___libc_free_0_0(v81);
          v81 = v93;
          v82 = v94;
          if ( (unsigned int)v92 > 0x40 && v91 )
            j_j___libc_free_0_0(v91);
        }
        else
        {
          v81 = v93;
          v82 = v94;
        }
      }
      v54 = *(_QWORD *)(v4 + 8);
      sub_ABB970((__int64)&v83, (__int64)&v75, (__int64)&v79);
      if ( !v87 )
      {
        v10 = *(_QWORD *)(a2 + 16);
        if ( !v10 )
          goto LABEL_40;
        v29 = *(_QWORD *)(v10 + 8);
        v10 = 0;
        if ( v29 )
          goto LABEL_40;
        v10 = *(_QWORD *)(a3 + 16);
        if ( !v10 )
          goto LABEL_40;
        if ( *(_QWORD *)(v10 + 8) )
        {
          v10 = 0;
LABEL_40:
          if ( v82 > 0x40 && v81 )
          {
            v57 = v10;
            j_j___libc_free_0_0(v81);
            v10 = v57;
          }
          if ( v80 > 0x40 && v79 )
          {
            v58 = v10;
            j_j___libc_free_0_0(v79);
            v10 = v58;
          }
          if ( v78 > 0x40 && v77 )
          {
            v59 = v10;
            j_j___libc_free_0_0(v77);
            v10 = v59;
          }
          if ( v76 > 0x40 && v75 )
          {
            v60 = v10;
            j_j___libc_free_0_0(v75);
            return v60;
          }
          return v10;
        }
        v35 = sub_AAFBB0((__int64)&v75);
        v10 = 0;
        if ( v35 )
          goto LABEL_64;
        v36 = sub_AAFBB0((__int64)&v79);
        v10 = 0;
        if ( v36 )
          goto LABEL_64;
        sub_9865C0((__int64)&v91, (__int64)&v75);
        sub_10B8490(&v91, (__int64 *)&v79);
        v68 = (int)v92;
        v67 = v91;
        sub_9865C0((__int64)&v88, (__int64)&v81);
        sub_C46F20((__int64)&v88, 1u);
        v37 = v89;
        v89 = 0;
        LODWORD(v92) = v37;
        v91 = (_QWORD **)v88;
        sub_9865C0((__int64)&v71, (__int64)&v77);
        sub_C46F20((__int64)&v71, 1u);
        v74 = v72;
        v73 = v71;
        v72 = 0;
        sub_10B8490(&v91, (__int64 *)&v73);
        v38 = (int)v92;
        LODWORD(v92) = 0;
        v70 = v38;
        v69 = v91;
        sub_969240((__int64 *)&v73);
        sub_969240((__int64 *)&v71);
        sub_969240((__int64 *)&v91);
        sub_969240((__int64 *)&v88);
        sub_9865C0((__int64)&v91, (__int64)&v77);
        sub_C46B40((__int64)&v91, (__int64 *)&v75);
        v72 = (unsigned int)v92;
        v71 = v91;
        if ( !sub_986BA0((__int64)&v67) )
          goto LABEL_115;
        if ( !sub_AAD8B0((__int64)&v67, &v69) )
          goto LABEL_115;
        sub_9865C0((__int64)&v88, (__int64)&v81);
        sub_C46B40((__int64)&v88, (__int64 *)&v79);
        v39 = v89;
        v89 = 0;
        LODWORD(v92) = v39;
        v91 = (_QWORD **)v88;
        v49 = sub_AAD8B0((__int64)&v71, &v91);
        sub_969240((__int64 *)&v91);
        sub_969240((__int64 *)&v88);
        if ( !v49 )
        {
LABEL_115:
          sub_969240((__int64 *)&v71);
          sub_969240((__int64 *)&v69);
          sub_969240((__int64 *)&v67);
          v24 = v87;
          v10 = 0;
LABEL_65:
          if ( v24 )
          {
            v87 = 0;
            if ( v86 > 0x40 && v85 )
            {
              v63 = v10;
              j_j___libc_free_0_0(v85);
              v10 = v63;
            }
            if ( v84 > 0x40 && v83 )
            {
              v64 = v10;
              j_j___libc_free_0_0(v83);
              v10 = v64;
            }
          }
          goto LABEL_40;
        }
        v40 = (__int64 *)&v75;
        if ( (int)sub_C49970((__int64)&v75, (unsigned __int64 *)&v79) >= 0 )
          v40 = (__int64 *)&v79;
        if ( v87 )
        {
          if ( v84 <= 0x40 && *((_DWORD *)v40 + 2) <= 0x40u )
          {
            v48 = *v40;
            v84 = *((_DWORD *)v40 + 2);
            v83 = v48;
          }
          else
          {
            sub_C43990((__int64)&v83, (__int64)v40);
          }
          if ( v86 <= 0x40 && *((_DWORD *)v40 + 6) <= 0x40u )
          {
            v47 = v40[2];
            v86 = *((_DWORD *)v40 + 6);
            v85 = v47;
          }
          else
          {
            sub_C43990((__int64)&v85, (__int64)(v40 + 2));
          }
        }
        else
        {
          sub_9865C0((__int64)&v83, (__int64)v40);
          sub_9865C0((__int64)&v85, (__int64)(v40 + 2));
          v87 = 1;
        }
        v41 = *(unsigned int ***)(a1 + 32);
        v95 = 257;
        sub_9865C0((__int64)&v73, (__int64)&v67);
        sub_987160((__int64)&v73, (__int64)&v67, v42, v43, v44);
        v45 = v74;
        v74 = 0;
        v89 = v45;
        v88 = v73;
        v46 = (_BYTE *)sub_AD8D80(v54, (__int64)&v88);
        v4 = sub_A82350(v41, (_BYTE *)v4, v46, (__int64)&v91);
        sub_969240((__int64 *)&v88);
        sub_969240((__int64 *)&v73);
        sub_969240((__int64 *)&v71);
        sub_969240((__int64 *)&v69);
        sub_969240((__int64 *)&v67);
      }
      if ( a4 )
      {
        sub_ABB300((__int64)&v91, (__int64)&v83);
        if ( v87 )
        {
          if ( v84 > 0x40 && v83 )
            j_j___libc_free_0_0(v83);
          v83 = (__int64)v91;
          v30 = (unsigned int)v92;
          LODWORD(v92) = 0;
          v84 = v30;
          if ( v86 > 0x40 && v85 )
          {
            j_j___libc_free_0_0(v85);
            v85 = v93;
            v86 = v94;
            if ( (unsigned int)v92 > 0x40 && v91 )
              j_j___libc_free_0_0(v91);
          }
          else
          {
            v85 = v93;
            v86 = v94;
          }
        }
        else
        {
          v87 = 1;
          v84 = (unsigned int)v92;
          v83 = (__int64)v91;
          v86 = v94;
          v85 = v93;
        }
      }
      v72 = 1;
      v71 = 0;
      v74 = 1;
      v73 = 0;
      sub_AAF830((__int64)&v83, (int *)&v69, (__int64)&v71, (__int64 *)&v73);
      v16 = v74;
      if ( v74 > 0x40 )
      {
        if ( v16 - (unsigned int)sub_C444A0((__int64)&v73) > 0x40 )
          goto LABEL_56;
        v17 = (_QWORD *)*v73;
      }
      else
      {
        v17 = v73;
      }
      if ( !v17 )
      {
LABEL_58:
        v21 = *(unsigned int ***)(a1 + 32);
        v95 = 257;
        v22 = (_BYTE *)sub_AD8D80(v54, (__int64)&v71);
        v23 = sub_92B530(v21, (unsigned int)v69, v4, v22, (__int64)&v91);
        v10 = v23;
        if ( v74 > 0x40 && v73 )
        {
          v61 = v23;
          j_j___libc_free_0_0(v73);
          v10 = v61;
        }
        if ( v72 > 0x40 && v71 )
        {
          v62 = v10;
          j_j___libc_free_0_0(v71);
          v10 = v62;
        }
LABEL_64:
        v24 = v87;
        goto LABEL_65;
      }
LABEL_56:
      v18 = *(__int64 **)(a1 + 32);
      v90 = 257;
      v19 = sub_AD8D80(v54, (__int64)&v73);
      v20 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v18[10] + 32LL))(
              v18[10],
              13,
              v4,
              v19,
              0,
              0);
      if ( !v20 )
      {
        v95 = 257;
        v20 = sub_B504D0(13, v4, v19, (__int64)&v91, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _QWORD **, __int64, __int64))(*(_QWORD *)v18[11] + 16LL))(
          v18[11],
          v20,
          &v88,
          v18[7],
          v18[8]);
        v31 = *v18;
        v32 = *v18 + 16LL * *((unsigned int *)v18 + 2);
        while ( v32 != v31 )
        {
          v33 = *(_QWORD *)(v31 + 8);
          v34 = *(_DWORD *)v31;
          v31 += 16;
          sub_B99FD0(v20, v34, v33);
        }
      }
      v4 = v20;
      goto LABEL_58;
    }
LABEL_28:
    v9 = sub_B52870(v51);
    goto LABEL_29;
  }
  return v10;
}
