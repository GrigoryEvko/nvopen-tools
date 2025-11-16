// Function: sub_1115C10
// Address: 0x1115c10
//
_QWORD *__fastcall sub_1115C10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r9
  __int64 v5; // r12
  unsigned __int16 v6; // bx
  __int64 v7; // rdi
  _BYTE *v8; // r15
  unsigned int v9; // ecx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned int v12; // esi
  unsigned __int64 v13; // rax
  bool v14; // zf
  unsigned int v15; // eax
  __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  unsigned int v18; // edx
  int v19; // r15d
  _QWORD *v20; // r13
  __int64 v21; // rdi
  __int64 v22; // r15
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rax
  int v26; // r14d
  __int64 v27; // r8
  __int16 v28; // bx
  __int64 v29; // rax
  __int64 v31; // r15
  __int64 v32; // rax
  _BYTE *v33; // rax
  bool v34; // al
  bool v35; // al
  unsigned int v36; // ebx
  unsigned __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // r15
  unsigned int v40; // eax
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned int v43; // r13d
  __int64 v44; // rbx
  unsigned int v45; // edx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rax
  _BYTE *v49; // rsi
  __int64 v50; // r12
  __int64 v51; // r15
  _QWORD *v52; // rax
  unsigned int v53; // edx
  unsigned __int64 v54; // rax
  __int64 v55; // r15
  _QWORD *v56; // rax
  int v57; // eax
  int v58; // eax
  __int64 v59; // r15
  _QWORD *v60; // rax
  __int64 v61; // r15
  _QWORD *v62; // rax
  __int64 v63; // [rsp+8h] [rbp-B8h]
  __int64 v64; // [rsp+8h] [rbp-B8h]
  __int64 v65; // [rsp+10h] [rbp-B0h]
  __int64 v66; // [rsp+10h] [rbp-B0h]
  __int64 v67; // [rsp+10h] [rbp-B0h]
  __int64 v68; // [rsp+10h] [rbp-B0h]
  __int64 v69; // [rsp+18h] [rbp-A8h]
  __int64 v70; // [rsp+18h] [rbp-A8h]
  __int64 v72; // [rsp+20h] [rbp-A0h]
  __int64 v73; // [rsp+20h] [rbp-A0h]
  _BYTE *v74; // [rsp+20h] [rbp-A0h]
  unsigned int v75; // [rsp+28h] [rbp-98h]
  __int64 v77; // [rsp+28h] [rbp-98h]
  __int64 v78; // [rsp+28h] [rbp-98h]
  __int64 v79; // [rsp+28h] [rbp-98h]
  __int64 v80; // [rsp+28h] [rbp-98h]
  __int64 v81; // [rsp+28h] [rbp-98h]
  __int64 v82; // [rsp+28h] [rbp-98h]
  __int64 v83; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v84; // [rsp+38h] [rbp-88h]
  unsigned __int64 v85; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v86; // [rsp+48h] [rbp-78h]
  unsigned __int64 v87; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v88; // [rsp+58h] [rbp-68h]
  unsigned __int64 v89; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v90; // [rsp+68h] [rbp-58h]
  __int16 v91; // [rsp+80h] [rbp-40h]

  v4 = a4;
  v5 = a3;
  v6 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( ((v6 - 34) & 0xFFFD) == 0 )
  {
    v7 = *(_QWORD *)(a3 - 32);
    v8 = (_BYTE *)(v7 + 24);
    if ( *(_BYTE *)v7 == 17 )
      goto LABEL_3;
    v24 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v24 <= 1 && *(_BYTE *)v7 <= 0x15u )
    {
      v25 = sub_AD7630(v7, 0, v24);
      if ( v25 )
      {
        if ( *v25 == 17 )
        {
          v4 = a4;
          v8 = v25 + 24;
LABEL_3:
          v9 = *(_DWORD *)(v4 + 8);
          v86 = v9;
          if ( v9 > 0x40 )
          {
            v80 = v4;
            sub_C43780((__int64)&v85, (const void **)v4);
            v4 = v80;
            if ( v6 != 36 )
              goto LABEL_65;
          }
          else
          {
            v85 = *(_QWORD *)v4;
            if ( v6 != 36 )
            {
              v10 = 1LL << ((unsigned __int8)v9 - 1);
              goto LABEL_6;
            }
          }
          v81 = v4;
          sub_C46E90((__int64)&v85);
          v4 = v81;
LABEL_65:
          v53 = *(_DWORD *)(v4 + 8);
          v10 = 1LL << ((unsigned __int8)v53 - 1);
          if ( v53 > 0x40 )
          {
            v11 = *(_QWORD *)(*(_QWORD *)v4 + 8LL * ((v53 - 1) >> 6));
LABEL_7:
            if ( (v10 & v11) != 0 )
            {
              if ( v86 > 0x40 )
              {
                sub_C43D10((__int64)&v85);
              }
              else
              {
                v54 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v86) & ~v85;
                if ( !v86 )
                  v54 = 0;
                v85 = v54;
              }
            }
            v12 = *((_DWORD *)v8 + 2);
            v13 = *(_QWORD *)v8;
            if ( v12 > 0x40 )
              v13 = *(_QWORD *)(v13 + 8LL * ((v12 - 1) >> 6));
            v14 = (v13 & (1LL << ((unsigned __int8)v12 - 1))) == 0;
            v15 = *((_DWORD *)v8 + 2);
            if ( v14 )
            {
              v88 = *((_DWORD *)v8 + 2);
              if ( v15 > 0x40 )
                sub_C43780((__int64)&v87, (const void **)v8);
              else
                v87 = *(_QWORD *)v8;
              goto LABEL_17;
            }
            v90 = *((_DWORD *)v8 + 2);
            if ( v15 > 0x40 )
            {
              sub_C43780((__int64)&v89, (const void **)v8);
              v15 = v90;
              if ( v90 > 0x40 )
              {
                sub_C43D10((__int64)&v89);
LABEL_16:
                sub_C46250((__int64)&v89);
                v88 = v90;
                v87 = v89;
LABEL_17:
                sub_C46F20((__int64)&v87, 1u);
                v18 = v88;
                v88 = 0;
                v90 = v18;
                v75 = v18;
                v89 = v87;
                v72 = v87;
                v19 = sub_C49970((__int64)&v85, &v89);
                if ( v75 > 0x40 )
                {
                  if ( v72 )
                  {
                    j_j___libc_free_0_0(v72);
                    if ( v88 > 0x40 )
                    {
                      if ( v87 )
                        j_j___libc_free_0_0(v87);
                    }
                  }
                }
                v20 = 0;
                if ( v19 >= 0 )
                {
                  v21 = *(_QWORD *)(v5 + 8);
                  if ( v6 == 34 )
                  {
                    v55 = sub_AD6530(v21, (__int64)&v89);
                    v91 = 257;
                    v56 = sub_BD2C40(72, unk_3F10FD0);
                    v20 = v56;
                    if ( v56 )
                      sub_1113300((__int64)v56, 40, v5, v55, (__int64)&v89);
                  }
                  else
                  {
                    v22 = sub_AD62B0(v21);
                    v91 = 257;
                    v23 = sub_BD2C40(72, unk_3F10FD0);
                    v20 = v23;
                    if ( v23 )
                      sub_1113300((__int64)v23, 38, v5, v22, (__int64)&v89);
                  }
                }
                if ( v86 > 0x40 && v85 )
                  j_j___libc_free_0_0(v85);
                return v20;
              }
              v16 = v89;
            }
            else
            {
              v16 = *(_QWORD *)v8;
            }
            v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & ~v16;
            if ( !v15 )
              v17 = 0;
            v89 = v17;
            goto LABEL_16;
          }
LABEL_6:
          v11 = *(_QWORD *)v4;
          goto LABEL_7;
        }
      }
    }
    return 0;
  }
  v26 = v6;
  v27 = a1;
  v28 = (v6 - 38) & 0xFFFD;
  if ( v28 && (unsigned int)(v26 - 32) > 1 )
    return 0;
  v29 = *(_QWORD *)(a3 + 16);
  if ( !v29 )
    return 0;
  v20 = *(_QWORD **)(v29 + 8);
  if ( v20 )
    return 0;
  v31 = *(_QWORD *)(a3 - 32);
  if ( *(_BYTE *)v31 == 17 )
  {
    v73 = v31 + 24;
    if ( *(_DWORD *)(v31 + 32) > 0x40u )
    {
      v57 = sub_C44630(v73);
      v27 = a1;
      v4 = a4;
      if ( v57 == 1 )
        goto LABEL_48;
    }
    else
    {
      v32 = *(_QWORD *)(v31 + 24);
      if ( v32 )
      {
        a3 = v32 - 1;
        if ( (v32 & (v32 - 1)) == 0 )
          goto LABEL_48;
      }
    }
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v31 + 8) + 8LL) - 17 > 1 )
      return v20;
  }
  else
  {
    a3 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v31 + 8) + 8LL) - 17;
    if ( (unsigned int)a3 > 1 || *(_BYTE *)v31 > 0x15u )
      return v20;
  }
  v69 = v4;
  v77 = v27;
  v33 = sub_AD7630(v31, 1, a3);
  if ( !v33 )
    return v20;
  if ( *v33 != 17 )
    return v20;
  v73 = (__int64)(v33 + 24);
  v34 = sub_986BA0((__int64)(v33 + 24));
  v27 = v77;
  v4 = v69;
  if ( !v34 )
    return v20;
LABEL_48:
  if ( !v28 )
  {
    v70 = v27;
    v78 = v4;
    v35 = sub_9867B0(v4);
    v4 = v78;
    v27 = v70;
    if ( !v35 )
      return 0;
  }
  if ( (unsigned int)(v26 - 32) <= 1 )
  {
    v36 = *(_DWORD *)(v4 + 8);
    v37 = *(_QWORD *)v4;
    v38 = 1LL << ((unsigned __int8)v36 - 1);
    if ( v36 > 0x40 )
    {
      if ( (*(_QWORD *)(v37 + 8LL * ((v36 - 1) >> 6)) & v38) != 0 )
        return 0;
      v68 = v27;
      v82 = v4;
      v58 = sub_C444A0(v4);
      v4 = v82;
      v27 = v68;
      if ( v36 == v58 )
        return 0;
    }
    else if ( (v38 & v37) != 0 || !v37 )
    {
      return 0;
    }
  }
  v39 = *(_QWORD *)(v5 + 8);
  v65 = v4;
  v79 = v27;
  v40 = sub_BCB060(v39);
  v41 = v79;
  v42 = v65;
  v43 = v40 - 1;
  v84 = v40;
  v44 = 1LL << ((unsigned __int8)v40 - 1);
  if ( v40 > 0x40 )
  {
    sub_C43690((__int64)&v83, 0, 0);
    v41 = v79;
    v42 = v65;
    if ( v84 > 0x40 )
    {
      *(_QWORD *)(v83 + 8LL * (v43 >> 6)) |= v44;
      goto LABEL_57;
    }
  }
  else
  {
    v83 = 0;
  }
  v83 |= v44;
LABEL_57:
  v63 = v42;
  v66 = v41;
  sub_9865C0((__int64)&v85, v73);
  sub_C46F20((__int64)&v85, 1u);
  v45 = v86;
  v86 = 0;
  v46 = v66;
  v47 = v63;
  v88 = v45;
  v87 = v85;
  if ( v45 > 0x40 )
  {
    sub_C43BD0(&v87, &v83);
    v45 = v88;
    v48 = v87;
    v47 = v63;
    v46 = v66;
  }
  else
  {
    v48 = v83 | v85;
    v87 = v83 | v85;
  }
  v90 = v45;
  v64 = v47;
  v67 = v46;
  v89 = v48;
  v88 = 0;
  v74 = (_BYTE *)sub_AD8D80(v39, (__int64)&v89);
  sub_969240((__int64 *)&v89);
  sub_969240((__int64 *)&v87);
  sub_969240((__int64 *)&v85);
  v91 = 257;
  v49 = *(_BYTE **)(v5 - 64);
  v50 = sub_A82350(*(unsigned int ***)(v67 + 32), v49, v74, (__int64)&v89);
  if ( (unsigned int)(v26 - 32) <= 1 )
  {
    v59 = sub_AD8D80(v39, v64);
    v91 = 257;
    v60 = sub_BD2C40(72, unk_3F10FD0);
    v20 = v60;
    if ( v60 )
      sub_1113300((__int64)v60, v26, v50, v59, (__int64)&v89);
  }
  else if ( v26 == 38 )
  {
    v61 = sub_AD6530(v39, (__int64)v49);
    v91 = 257;
    v62 = sub_BD2C40(72, unk_3F10FD0);
    v20 = v62;
    if ( v62 )
      sub_1113300((__int64)v62, 38, v50, v61, (__int64)&v89);
  }
  else
  {
    v51 = sub_AD8D80(v39, (__int64)&v83);
    v91 = 257;
    v52 = sub_BD2C40(72, unk_3F10FD0);
    v20 = v52;
    if ( v52 )
      sub_1113300((__int64)v52, 34, v50, v51, (__int64)&v89);
  }
  sub_969240(&v83);
  return v20;
}
