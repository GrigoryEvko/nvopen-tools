// Function: sub_17D4DA0
// Address: 0x17d4da0
//
__int64 *__fastcall sub_17D4DA0(__int128 a1)
{
  __int64 *v1; // r12
  unsigned __int8 v2; // al
  __int64 *result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 **v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rsi
  __int64 *v9; // rax
  __int64 *v10; // r14
  int v11; // r15d
  __int64 *v12; // r13
  unsigned int v13; // r10d
  unsigned int v14; // ebx
  __int64 v15; // rax
  __int64 v16; // r11
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 **v20; // rax
  __int64 v21; // r11
  unsigned int v22; // r10d
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // rbx
  __int64 v32; // rax
  __int64 v33; // r10
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rax
  _QWORD *v37; // rax
  unsigned int v38; // eax
  unsigned int v39; // r10d
  _QWORD *v40; // r11
  unsigned int v41; // r8d
  __int64 *v42; // rax
  int v43; // r8d
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned int v48; // r10d
  __int64 v49; // r11
  __int64 *v50; // rbx
  __int64 v51; // rax
  unsigned __int64 v52; // rcx
  __int64 **v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rdi
  __int64 *v57; // rax
  __int64 v58; // rax
  unsigned int v59; // eax
  __int64 v60; // rax
  __int64 v61; // rdi
  __int64 v62; // r10
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rsi
  __int64 v66; // rdx
  unsigned __int8 *v67; // rsi
  _QWORD *v68; // [rsp+8h] [rbp-128h]
  unsigned int v69; // [rsp+14h] [rbp-11Ch]
  __int64 v70; // [rsp+18h] [rbp-118h]
  __int64 v71; // [rsp+18h] [rbp-118h]
  __int64 v72; // [rsp+18h] [rbp-118h]
  unsigned int v73; // [rsp+18h] [rbp-118h]
  unsigned int v74; // [rsp+18h] [rbp-118h]
  unsigned int v75; // [rsp+18h] [rbp-118h]
  __int64 *v76; // [rsp+18h] [rbp-118h]
  __int64 v77; // [rsp+18h] [rbp-118h]
  unsigned int v78; // [rsp+20h] [rbp-110h]
  unsigned int v79; // [rsp+20h] [rbp-110h]
  unsigned int v80; // [rsp+20h] [rbp-110h]
  unsigned int v81; // [rsp+20h] [rbp-110h]
  __int64 v82; // [rsp+20h] [rbp-110h]
  unsigned int v83; // [rsp+20h] [rbp-110h]
  _QWORD *v84; // [rsp+20h] [rbp-110h]
  __int64 v85; // [rsp+20h] [rbp-110h]
  __int64 v86; // [rsp+20h] [rbp-110h]
  __int64 v87; // [rsp+20h] [rbp-110h]
  __int64 v88; // [rsp+20h] [rbp-110h]
  __int64 v89; // [rsp+20h] [rbp-110h]
  __int64 v90; // [rsp+20h] [rbp-110h]
  __int64 v91; // [rsp+20h] [rbp-110h]
  __int64 *v92; // [rsp+38h] [rbp-F8h]
  __int64 *v93; // [rsp+40h] [rbp-F0h]
  unsigned int v94; // [rsp+40h] [rbp-F0h]
  unsigned int v95; // [rsp+40h] [rbp-F0h]
  __int64 v96; // [rsp+40h] [rbp-F0h]
  _QWORD *v97; // [rsp+50h] [rbp-E0h]
  __int64 v98; // [rsp+58h] [rbp-D8h]
  __int64 *v99; // [rsp+58h] [rbp-D8h]
  __int64 v100; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v101[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v102; // [rsp+80h] [rbp-B0h]
  __int64 v103[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int16 v104; // [rsp+A0h] [rbp-90h]
  __int64 v105; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v106; // [rsp+B8h] [rbp-78h]
  __int64 *v107; // [rsp+C0h] [rbp-70h]
  _QWORD *v108; // [rsp+C8h] [rbp-68h]

  v1 = (__int64 *)*((_QWORD *)&a1 + 1);
  if ( !*(_BYTE *)(a1 + 489) )
    goto LABEL_5;
  v2 = *(_BYTE *)(*((_QWORD *)&a1 + 1) + 16LL);
  if ( v2 <= 0x17u )
  {
    if ( v2 == 9 )
    {
      *((_QWORD *)&a1 + 1) = **((_QWORD **)&a1 + 1);
      if ( !*(_BYTE *)(a1 + 491) )
        return (__int64 *)sub_17CDAE0((_QWORD *)a1, *((__int64 *)&a1 + 1));
      result = sub_17CD8D0((_QWORD *)a1, *((__int64 *)&a1 + 1));
      if ( result )
      {
        *(_QWORD *)&a1 = result;
        return (__int64 *)sub_17CCDD0(a1, v4, v5);
      }
      return result;
    }
    if ( v2 == 17 )
    {
      v6 = (__int64 **)sub_17D46A0(a1 + 304, *((__int64 *)&a1 + 1));
      result = *v6;
      if ( *v6 )
        return result;
      v7 = *(_QWORD *)(*((_QWORD *)&a1 + 1) + 24LL);
      v8 = sub_157ED20(*(_QWORD *)(a1 + 480));
      sub_17CE510((__int64)&v105, v8, 0, 0, 0);
      v98 = sub_1632FA0(*(_QWORD *)(v7 + 40));
      if ( (*(_BYTE *)(v7 + 18) & 1) != 0 )
      {
        sub_15E08E0(v7, v8);
        v9 = *(__int64 **)(v7 + 88);
        v93 = &v9[5 * *(_QWORD *)(v7 + 96)];
        if ( (*(_BYTE *)(v7 + 18) & 1) != 0 )
        {
          sub_15E08E0(v7, v8);
          v9 = *(__int64 **)(v7 + 88);
        }
      }
      else
      {
        v9 = *(__int64 **)(v7 + 88);
        v93 = &v9[5 * *(_QWORD *)(v7 + 96)];
      }
      v10 = v9;
      if ( v9 == v93 )
      {
LABEL_28:
        v99 = *v6;
        sub_17CD270(&v105);
        return v99;
      }
      v92 = (__int64 *)v6;
      v11 = 0;
      v97 = (_QWORD *)a1;
      v12 = v93;
      while ( 1 )
      {
        if ( !sub_1704BC0(*v10, 0) )
          goto LABEL_23;
        if ( (unsigned __int8)sub_15E0450((__int64)v10) )
          v13 = sub_12BE0A0(v98, **(_QWORD **)(*v10 + 16));
        else
          v13 = sub_12BE0A0(v98, *v10);
        v14 = v13;
        if ( v10 == v1 )
        {
          v94 = v13 + v11;
          v102 = 257;
          v15 = v97[1];
          v16 = *(_QWORD *)(v15 + 192);
          v17 = *(_QWORD *)(v15 + 176);
          if ( v17 != *(_QWORD *)v16 )
          {
            v78 = v13;
            if ( *(_BYTE *)(v16 + 16) > 0x10u )
            {
              v104 = 257;
              v47 = sub_15FDFF0(v16, v17, (__int64)v103, 0);
              v48 = v78;
              v49 = v47;
              if ( v106 )
              {
                v50 = v107;
                v74 = v78;
                v85 = v47;
                sub_157E9D0(v106 + 40, v47);
                v49 = v85;
                v48 = v74;
                v51 = *(_QWORD *)(v85 + 24);
                v52 = *v50 & 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v85 + 32) = v50;
                *(_QWORD *)(v85 + 24) = v52 | v51 & 7;
                *(_QWORD *)(v52 + 8) = v85 + 24;
                *v50 = *v50 & 7 | (v85 + 24);
              }
              v75 = v48;
              v86 = v49;
              sub_164B780(v49, v101);
              sub_12A86E0(&v105, v86);
              v13 = v75;
              v16 = v86;
            }
            else
            {
              v18 = sub_15A4A70((__int64 ***)v16, v17);
              v13 = v78;
              v16 = v18;
            }
          }
          if ( v11 )
          {
            v104 = 257;
            v81 = v13;
            v72 = v16;
            v29 = sub_15A0680(*(_QWORD *)(v97[1] + 176LL), v11, 0);
            v30 = sub_12899C0(&v105, v72, v29, (__int64)v103, 0, 0);
            v13 = v81;
            v16 = v30;
          }
          v103[0] = (__int64)"_msarg";
          v104 = 259;
          v70 = v16;
          v79 = v13;
          v19 = sub_17CD8D0(v97, *v10);
          v20 = (__int64 **)sub_1646BA0(v19, 0);
          v21 = v70;
          v22 = v79;
          if ( v20 != *(__int64 ***)v70 )
          {
            if ( *(_BYTE *)(v70 + 16) > 0x10u )
              v23 = (__int64)sub_17CD020(&v105, 46, v70, (__int64)v20, v103);
            else
              v23 = sub_15A46C0(46, (__int64 ***)v70, v20, 0);
            v22 = v79;
            v21 = v23;
          }
          v71 = v21;
          v80 = v22;
          if ( (unsigned __int8)sub_15E0450((__int64)v10) )
          {
            v38 = sub_15E0370((__int64)v10);
            v39 = v80;
            v40 = (_QWORD *)v71;
            v41 = v38;
            if ( !v38 )
            {
              v59 = sub_15A9FE0(v98, **(_QWORD **)(*v1 + 16));
              v40 = (_QWORD *)v71;
              v39 = v80;
              v41 = v59;
            }
            v68 = v40;
            v69 = v39;
            v83 = v41;
            v42 = (__int64 *)sub_1643330(v108);
            v73 = v83;
            v14 = v69;
            v84 = (_QWORD *)sub_17CFB40((__int64)v97, (__int64)v1, &v105, v42, v83);
            v43 = v73;
            if ( v94 > 0x320 )
            {
              v53 = (__int64 **)sub_1643330(v108);
              v96 = sub_15A06D0(v53, (__int64)v1, v54, v55);
              v56 = sub_1643360(v108);
              v57 = (__int64 *)sub_159C470(v56, v69, 0);
              sub_15E7280(&v105, v84, v96, v57, v73, 0, 0, 0, 0);
              v24 = *v1;
              v58 = sub_17CDAE0(v97, *v1);
              v26 = (__int64)v92;
              *v92 = v58;
              v27 = v97[1];
            }
            else
            {
              if ( v73 > 8 )
                v43 = 8;
              v95 = v43;
              v44 = sub_1643360(v108);
              v45 = (__int64 *)sub_159C470(v44, v69, 0);
              sub_15E7430(&v105, v84, v95, v68, v95, v45, 0, 0, 0, 0, 0);
              v24 = *v1;
              v46 = sub_17CDAE0(v97, *v1);
              v26 = (__int64)v92;
              *v92 = v46;
              v32 = v97[1];
              v27 = v32;
              if ( *(_DWORD *)(v32 + 156) )
              {
LABEL_45:
                v102 = 257;
                v33 = *(_QWORD *)(v32 + 200);
                v34 = *(_QWORD *)(v32 + 176);
                if ( v34 != *(_QWORD *)v33 )
                {
                  if ( *(_BYTE *)(v33 + 16) > 0x10u )
                  {
                    v61 = *(_QWORD *)(v32 + 200);
                    v104 = 257;
                    v62 = sub_15FDFF0(v61, v34, (__int64)v103, 0);
                    if ( v106 )
                    {
                      v88 = v62;
                      v76 = v107;
                      sub_157E9D0(v106 + 40, v62);
                      v62 = v88;
                      v63 = *(_QWORD *)(v88 + 24);
                      v64 = *v76;
                      *(_QWORD *)(v88 + 32) = v76;
                      v64 &= 0xFFFFFFFFFFFFFFF8LL;
                      *(_QWORD *)(v88 + 24) = v64 | v63 & 7;
                      *(_QWORD *)(v64 + 8) = v88 + 24;
                      *v76 = *v76 & 7 | (v88 + 24);
                    }
                    v89 = v62;
                    sub_164B780(v62, v101);
                    v33 = v89;
                    if ( !v105 )
                      goto LABEL_70;
                    v100 = v105;
                    sub_1623A60((__int64)&v100, v105, 2);
                    v33 = v89;
                    v65 = *(_QWORD *)(v89 + 48);
                    v66 = v89 + 48;
                    if ( v65 )
                    {
                      v77 = v89;
                      v90 = v89 + 48;
                      sub_161E7C0(v90, v65);
                      v33 = v77;
                      v66 = v90;
                    }
                    v67 = (unsigned __int8 *)v100;
                    *(_QWORD *)(v33 + 48) = v100;
                    if ( v67 )
                    {
                      v91 = v33;
                      sub_1623210((__int64)&v100, v67, v66);
                      v33 = v91;
                      v32 = v97[1];
                    }
                    else
                    {
LABEL_70:
                      v32 = v97[1];
                    }
                  }
                  else
                  {
                    v33 = sub_15A4A70(*(__int64 ****)(v32 + 200), v34);
                    v32 = v97[1];
                  }
                }
                if ( v11 )
                {
                  v104 = 257;
                  v87 = v33;
                  v60 = sub_15A0680(*(_QWORD *)(v32 + 176), v11, 0);
                  v33 = sub_12899C0(&v105, v87, v60, (__int64)v103, 0, 0);
                  v32 = v97[1];
                }
                v103[0] = (__int64)"_msarg_o";
                v104 = 259;
                v82 = v33;
                v35 = sub_1646BA0(*(__int64 **)(v32 + 184), 0);
                v36 = sub_12AA3B0(&v105, 0x2Eu, v82, v35, (__int64)v103);
                v104 = 257;
                v37 = sub_156E5B0(&v105, v36, (__int64)v103);
                sub_17D4B80((__int64)v97, (__int64)v1, (__int64)v37);
                goto LABEL_22;
              }
            }
          }
          else if ( v94 <= 0x320 )
          {
            v104 = 257;
            v24 = 8;
            v31 = sub_156E5B0(&v105, v71, (__int64)v103);
            sub_15F8F50((__int64)v31, 8u);
            *v92 = (__int64)v31;
            v14 = v80;
            v32 = v97[1];
            v26 = *(unsigned int *)(v32 + 156);
            v27 = v32;
            if ( (_DWORD)v26 )
              goto LABEL_45;
          }
          else
          {
            v24 = *v1;
            v25 = sub_17CDAE0(v97, *v1);
            v26 = (__int64)v92;
            *v92 = v25;
            v27 = v97[1];
            v14 = v80;
          }
          v28 = sub_15A06D0(*(__int64 ***)(v27 + 184), v24, v27, v26);
          sub_17D4B80((__int64)v97, (__int64)v1, v28);
        }
LABEL_22:
        v11 += (v14 + 7) & 0xFFFFFFF8;
LABEL_23:
        v10 += 5;
        if ( v12 == v10 )
        {
          v6 = (__int64 **)v92;
          goto LABEL_28;
        }
      }
    }
LABEL_5:
    *((_QWORD *)&a1 + 1) = **((_QWORD **)&a1 + 1);
    return (__int64 *)sub_17CDAE0((_QWORD *)a1, *((__int64 *)&a1 + 1));
  }
  if ( (*(_QWORD *)(*((_QWORD *)&a1 + 1) + 48LL) || *(__int16 *)(*((_QWORD *)&a1 + 1) + 18LL) < 0)
    && sub_1625940(*((__int64 *)&a1 + 1), "nosanitize", 0xAu) )
  {
    goto LABEL_5;
  }
  return (__int64 *)*sub_17D46A0(a1 + 304, *((__int64 *)&a1 + 1));
}
