// Function: sub_1961B00
// Address: 0x1961b00
//
__int64 __fastcall sub_1961B00(__int64 a1)
{
  __int64 result; // rax
  int v2; // edx
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // rsi
  __int64 v10; // rdi
  int v11; // eax
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rdx
  _QWORD *v16; // r10
  __int64 v17; // rax
  __int64 v18; // rsi
  int v19; // eax
  unsigned int v20; // ecx
  __int64 *v21; // rdx
  __int64 v22; // r9
  __int64 v23; // rdi
  _BOOL4 v24; // eax
  __int64 v25; // r15
  _QWORD *v26; // rax
  __int64 v27; // r14
  __int16 v28; // ax
  __int64 *v29; // r13
  __int64 v30; // rsi
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 v33; // r9
  const char *v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdx
  int v37; // ebx
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // rbx
  char *v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r9
  __int64 v44; // r8
  __int64 v45; // rsi
  __int64 *v46; // rbx
  __int64 v47; // rcx
  __int64 *v48; // rax
  __int64 v49; // rdi
  unsigned __int64 v50; // rcx
  __int64 v51; // rcx
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rdx
  int v56; // eax
  __int64 v57; // rax
  int v58; // ecx
  __int64 v59; // r9
  const char *v60; // rax
  __int64 v61; // rdi
  __int64 v62; // rdx
  int v63; // r15d
  __int64 v64; // rax
  __int64 *v65; // r10
  __int64 v66; // rbx
  __int64 v67; // r15
  char *v68; // rax
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // rcx
  __int64 *v72; // r14
  __int64 v73; // rsi
  __int64 *v74; // r12
  __int64 v75; // rdi
  __int64 **v76; // rax
  __int64 *v77; // r8
  unsigned __int64 v78; // rdi
  __int64 v79; // rdi
  __int64 v80; // rdi
  __int64 v81; // rdx
  int v82; // eax
  __int64 v83; // rax
  int v84; // edi
  int v85; // edx
  int v86; // edx
  int v87; // r8d
  int v88; // r8d
  __int64 v89; // [rsp-A0h] [rbp-A0h]
  __int64 v90; // [rsp-A0h] [rbp-A0h]
  __int64 v91; // [rsp-98h] [rbp-98h]
  __int64 v92; // [rsp-98h] [rbp-98h]
  __int64 *v93; // [rsp-98h] [rbp-98h]
  __int64 *v94; // [rsp-90h] [rbp-90h]
  __int64 v95; // [rsp-88h] [rbp-88h]
  __int64 *v96; // [rsp-80h] [rbp-80h]
  __int64 v97; // [rsp-80h] [rbp-80h]
  __int64 v98; // [rsp-80h] [rbp-80h]
  __int64 v99; // [rsp-80h] [rbp-80h]
  __int64 v100; // [rsp-80h] [rbp-80h]
  __int64 v101; // [rsp-80h] [rbp-80h]
  __int64 v102; // [rsp-80h] [rbp-80h]
  __int64 i; // [rsp-70h] [rbp-70h]
  const char *v104; // [rsp-68h] [rbp-68h] BYREF
  __int64 v105; // [rsp-60h] [rbp-60h]
  unsigned __int8 *v106; // [rsp-58h] [rbp-58h] BYREF
  const char *v107; // [rsp-50h] [rbp-50h]
  __int16 v108; // [rsp-48h] [rbp-48h]

  result = *(_QWORD *)(a1 + 32);
  v2 = *(_DWORD *)(result + 8);
  if ( v2 )
  {
    v3 = a1;
    v94 = (__int64 *)(a1 + 88);
    v95 = 8LL * (unsigned int)(v2 - 1);
    for ( i = 0; ; i = v4 + 8 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)result + i);
      v6 = sub_1B40B40(*(_QWORD *)(v3 + 8));
      if ( *(_BYTE *)(v6 + 16) > 0x17u )
      {
        v7 = *(_QWORD *)(v3 + 64);
        v8 = *(_DWORD *)(v7 + 24);
        if ( !v8 )
        {
          v16 = *(_QWORD **)(v3 + 16);
          goto LABEL_20;
        }
        v9 = *(_QWORD *)(v6 + 40);
        v10 = *(_QWORD *)(v7 + 8);
        v11 = v8 - 1;
        v12 = v11 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( v9 != *v13 )
        {
          v85 = 1;
          while ( v14 != -8 )
          {
            v88 = v85 + 1;
            v12 = v11 & (v85 + v12);
            v13 = (__int64 *)(v10 + 16LL * v12);
            v14 = *v13;
            if ( v9 == *v13 )
              goto LABEL_12;
            v85 = v88;
          }
LABEL_82:
          v16 = *(_QWORD **)(v3 + 16);
          if ( *((_BYTE *)v16 + 16) <= 0x17u )
            goto LABEL_20;
          v18 = v16[5];
          goto LABEL_17;
        }
LABEL_12:
        v15 = v13[1];
        if ( !v15 )
          goto LABEL_82;
        if ( !sub_1377F70(v15 + 56, v5) )
        {
          v33 = *(_QWORD *)(v5 + 48);
          if ( v33 )
            v33 -= 24;
          v91 = v33;
          v34 = sub_1649960(v6);
          v35 = *(_QWORD *)(v3 + 48);
          v104 = v34;
          v105 = v36;
          v106 = (unsigned __int8 *)&v104;
          v108 = 773;
          v107 = ".lcssa";
          v37 = sub_1961780(v35, v5);
          v98 = *(_QWORD *)v6;
          v38 = sub_1648B60(64);
          v39 = v38;
          if ( v38 )
          {
            sub_15F1EA0(v38, v98, 53, 0, 0, v91);
            *(_DWORD *)(v39 + 56) = v37;
            sub_164B780(v39, (__int64 *)&v106);
            sub_1648880(v39, *(_DWORD *)(v39 + 56), 1);
          }
          v99 = *(_QWORD *)(v3 + 48);
          v40 = (unsigned int)sub_1961780(v99, v5);
          v41 = sub_1415970(v99, v5);
          v44 = v6 + 8;
          v45 = (__int64)&v41[8 * v40];
          v46 = (__int64 *)v41;
          if ( (char *)v45 != v41 )
          {
            do
            {
              v55 = *v46;
              v56 = *(_DWORD *)(v39 + 20) & 0xFFFFFFF;
              if ( v56 == *(_DWORD *)(v39 + 56) )
              {
                v100 = *v46;
                sub_15F55D0(v39, v45, v55, v42, v44, v43);
                v55 = v100;
                v56 = *(_DWORD *)(v39 + 20) & 0xFFFFFFF;
              }
              v57 = (v56 + 1) & 0xFFFFFFF;
              v58 = v57 | *(_DWORD *)(v39 + 20) & 0xF0000000;
              *(_DWORD *)(v39 + 20) = v58;
              if ( (v58 & 0x40000000) != 0 )
                v47 = *(_QWORD *)(v39 - 8);
              else
                v47 = v39 - 24 * v57;
              v48 = (__int64 *)(v47 + 24LL * (unsigned int)(v57 - 1));
              if ( *v48 )
              {
                v49 = v48[1];
                v50 = v48[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v50 = v49;
                if ( v49 )
                {
                  v43 = *(_QWORD *)(v49 + 16) & 3LL;
                  *(_QWORD *)(v49 + 16) = v43 | v50;
                }
              }
              *v48 = v6;
              v51 = *(_QWORD *)(v6 + 8);
              v48[1] = v51;
              if ( v51 )
              {
                v43 = (__int64)(v48 + 1);
                *(_QWORD *)(v51 + 16) = (unsigned __int64)(v48 + 1) | *(_QWORD *)(v51 + 16) & 3LL;
              }
              v48[2] = (v6 + 8) | v48[2] & 3;
              *(_QWORD *)(v6 + 8) = v48;
              v52 = *(_DWORD *)(v39 + 20) & 0xFFFFFFF;
              v53 = (unsigned int)(v52 - 1);
              if ( (*(_BYTE *)(v39 + 23) & 0x40) != 0 )
                v54 = *(_QWORD *)(v39 - 8);
              else
                v54 = v39 - 24 * v52;
              ++v46;
              v42 = 3LL * *(unsigned int *)(v39 + 56);
              *(_QWORD *)(v54 + 8 * v53 + 24LL * *(unsigned int *)(v39 + 56) + 8) = v55;
            }
            while ( (__int64 *)v45 != v46 );
          }
          v6 = v39;
        }
      }
      v16 = *(_QWORD **)(v3 + 16);
      if ( *((_BYTE *)v16 + 16) <= 0x17u )
        goto LABEL_20;
      v17 = *(_QWORD *)(v3 + 64);
      v18 = v16[5];
      v10 = *(_QWORD *)(v17 + 8);
      v19 = *(_DWORD *)(v17 + 24);
      if ( !v19 )
        goto LABEL_20;
      v11 = v19 - 1;
LABEL_17:
      v20 = v11 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v21 = (__int64 *)(v10 + 16LL * v20);
      v22 = *v21;
      if ( *v21 == v18 )
      {
LABEL_18:
        v23 = v21[1];
        if ( v23 )
        {
          v96 = v16;
          v24 = sub_1377F70(v23 + 56, v5);
          v16 = v96;
          if ( !v24 )
          {
            v59 = *(_QWORD *)(v5 + 48);
            if ( v59 )
              v59 -= 24;
            v89 = v59;
            v60 = sub_1649960((__int64)v96);
            v61 = *(_QWORD *)(v3 + 48);
            v104 = v60;
            v108 = 773;
            v106 = (unsigned __int8 *)&v104;
            v105 = v62;
            v107 = ".lcssa";
            v63 = sub_1961780(v61, v5);
            v92 = *v96;
            v64 = sub_1648B60(64);
            v65 = v96;
            v66 = v64;
            if ( v64 )
            {
              sub_15F1EA0(v64, v92, 53, 0, 0, v89);
              *(_DWORD *)(v66 + 56) = v63;
              sub_164B780(v66, (__int64 *)&v106);
              sub_1648880(v66, *(_DWORD *)(v66 + 56), 1);
              v65 = v96;
            }
            v93 = v65;
            v101 = *(_QWORD *)(v3 + 48);
            v67 = (unsigned int)sub_1961780(v101, v5);
            v68 = sub_1415970(v101, v5);
            v71 = (__int64)&v68[8 * v67];
            v72 = (__int64 *)v68;
            v73 = (__int64)(v93 + 1);
            if ( (char *)v71 != v68 )
            {
              v90 = v3;
              v74 = (__int64 *)&v68[8 * v67];
              do
              {
                v81 = *v72;
                v82 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
                if ( v82 == *(_DWORD *)(v66 + 56) )
                {
                  v102 = *v72;
                  sub_15F55D0(v66, v73, v81, v71, v69, v70);
                  v81 = v102;
                  v82 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
                }
                v83 = (v82 + 1) & 0xFFFFFFF;
                v84 = v83 | *(_DWORD *)(v66 + 20) & 0xF0000000;
                *(_DWORD *)(v66 + 20) = v84;
                if ( (v84 & 0x40000000) != 0 )
                  v75 = *(_QWORD *)(v66 - 8);
                else
                  v75 = v66 - 24 * v83;
                v76 = (__int64 **)(v75 + 24LL * (unsigned int)(v83 - 1));
                if ( *v76 )
                {
                  v77 = v76[1];
                  v78 = (unsigned __int64)v76[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v78 = v77;
                  if ( v77 )
                  {
                    v70 = v77[2] & 3;
                    v77[2] = v70 | v78;
                  }
                }
                *v76 = v93;
                v79 = v93[1];
                v76[1] = (__int64 *)v79;
                if ( v79 )
                {
                  v70 = (__int64)(v76 + 1);
                  *(_QWORD *)(v79 + 16) = (unsigned __int64)(v76 + 1) | *(_QWORD *)(v79 + 16) & 3LL;
                }
                v76[2] = (__int64 *)(v73 | (unsigned __int64)v76[2] & 3);
                v93[1] = (__int64)v76;
                v80 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
                if ( (*(_BYTE *)(v66 + 23) & 0x40) != 0 )
                  v69 = *(_QWORD *)(v66 - 8);
                else
                  v69 = v66 - 24 * v80;
                ++v72;
                *(_QWORD *)(v69 + 8LL * (unsigned int)(v80 - 1) + 24LL * *(unsigned int *)(v66 + 56) + 8) = v81;
              }
              while ( v74 != v72 );
              v3 = v90;
            }
            v16 = (_QWORD *)v66;
          }
        }
      }
      else
      {
        v86 = 1;
        while ( v22 != -8 )
        {
          v87 = v86 + 1;
          v20 = v11 & (v86 + v20);
          v21 = (__int64 *)(v10 + 16LL * v20);
          v22 = *v21;
          if ( *v21 == v18 )
            goto LABEL_18;
          v86 = v87;
        }
      }
LABEL_20:
      v97 = (__int64)v16;
      v25 = *(_QWORD *)(**(_QWORD **)(v3 + 40) + i);
      v26 = sub_1648A60(64, 2u);
      v27 = (__int64)v26;
      if ( v26 )
        sub_15F9660((__int64)v26, v6, v97, v25);
      if ( *(_BYTE *)(v3 + 84) )
      {
        v28 = *(_WORD *)(v27 + 18) & 0xFC7F;
        LOBYTE(v28) = v28 | 0x80;
        *(_WORD *)(v27 + 18) = v28;
      }
      v29 = (__int64 *)(v27 + 48);
      sub_15F9450(v27, *(_DWORD *)(v3 + 80));
      v30 = *(_QWORD *)(v3 + 72);
      v106 = (unsigned __int8 *)v30;
      if ( !v30 )
      {
        if ( v29 == (__int64 *)&v106 )
          goto LABEL_6;
        v31 = *(_QWORD *)(v27 + 48);
        if ( !v31 )
          goto LABEL_6;
LABEL_27:
        sub_161E7C0(v27 + 48, v31);
        goto LABEL_28;
      }
      sub_1623A60((__int64)&v106, v30, 2);
      if ( v29 == (__int64 *)&v106 )
      {
        if ( v106 )
          sub_161E7C0((__int64)&v106, (__int64)v106);
LABEL_6:
        if ( *(_QWORD *)(v3 + 88) )
          goto LABEL_7;
        goto LABEL_30;
      }
      v31 = *(_QWORD *)(v27 + 48);
      if ( v31 )
        goto LABEL_27;
LABEL_28:
      v32 = v106;
      *(_QWORD *)(v27 + 48) = v106;
      if ( !v32 )
        goto LABEL_6;
      sub_1623210((__int64)&v106, v32, v27 + 48);
      if ( *(_QWORD *)(v3 + 88) )
        goto LABEL_7;
LABEL_30:
      if ( *(_QWORD *)(v3 + 96) || *(_QWORD *)(v3 + 104) )
      {
LABEL_7:
        sub_1626170(v27, v94);
        v4 = i;
        result = v95;
        if ( i == v95 )
          return result;
        goto LABEL_8;
      }
      v4 = i;
      result = v95;
      if ( i == v95 )
        return result;
LABEL_8:
      result = *(_QWORD *)(v3 + 32);
    }
  }
  return result;
}
