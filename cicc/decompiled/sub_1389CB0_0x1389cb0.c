// Function: sub_1389CB0
// Address: 0x1389cb0
//
void __fastcall sub_1389CB0(__int64 a1, __int64 a2)
{
  char *v2; // r14
  __int64 v3; // rbx
  char v4; // dl
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r10
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r12
  int v17; // r12d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r15
  char *v21; // rbx
  __int64 v22; // r12
  unsigned __int8 v23; // al
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r15
  unsigned __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 *v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r13
  unsigned __int64 v46; // r15
  unsigned __int64 v47; // r12
  unsigned __int64 v48; // r14
  __int64 v49; // r13
  __int64 v50; // r15
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rsi
  unsigned int v55; // edi
  __int64 *v56; // rax
  __int64 v57; // r10
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rbx
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // rsi
  unsigned int v68; // edi
  char **v69; // rax
  char *v70; // r10
  char *v71; // rdx
  int v72; // eax
  int v73; // r8d
  __int64 v74; // rax
  int v75; // eax
  int v76; // r8d
  unsigned __int64 v77; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v78; // [rsp+10h] [rbp-A0h]
  __int64 v79; // [rsp+18h] [rbp-98h]
  __int64 v80; // [rsp+20h] [rbp-90h]
  __int64 v81; // [rsp+20h] [rbp-90h]
  __int64 v82; // [rsp+20h] [rbp-90h]
  __int64 v83; // [rsp+28h] [rbp-88h]
  __int64 v84; // [rsp+28h] [rbp-88h]
  __int64 v85; // [rsp+28h] [rbp-88h]
  char *v86; // [rsp+28h] [rbp-88h]
  __int64 v87; // [rsp+30h] [rbp-80h]
  _QWORD v88[2]; // [rsp+38h] [rbp-78h] BYREF
  __int64 v89; // [rsp+48h] [rbp-68h] BYREF
  _QWORD *v90; // [rsp+50h] [rbp-60h] BYREF
  __int64 v91; // [rsp+58h] [rbp-58h]
  _QWORD v92[10]; // [rsp+60h] [rbp-50h] BYREF

  v2 = (char *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v3 = a1;
  v88[0] = a2;
  v4 = *(_BYTE *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  if ( (a2 & 4) == 0 )
  {
    if ( v4 >= 0 )
    {
      v13 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      v12 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      v11 = -72;
      goto LABEL_15;
    }
    v14 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
    v16 = v14 + v15;
    if ( v2[23] >= 0 )
    {
      if ( (unsigned int)(v16 >> 4) )
        goto LABEL_107;
    }
    else if ( (unsigned int)((v16 - sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL)) >> 4) )
    {
      if ( v2[23] < 0 )
      {
        v17 = *(_DWORD *)(sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL) + 8);
        if ( v2[23] < 0 )
        {
          v18 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
          v11 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v18 + v19 - 4) - v17);
          v12 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
          v13 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
          goto LABEL_15;
        }
LABEL_108:
        BUG();
      }
      goto LABEL_107;
    }
    v11 = -72;
    v12 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
    v13 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_15;
  }
  if ( v4 >= 0 )
  {
    v13 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    v11 = -24;
    goto LABEL_15;
  }
  v5 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v7 = v5 + v6;
  if ( v2[23] >= 0 )
  {
    if ( !(unsigned int)(v7 >> 4) )
      goto LABEL_30;
LABEL_107:
    BUG();
  }
  if ( !(unsigned int)((v7 - sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL)) >> 4) )
  {
LABEL_30:
    v11 = -24;
    v12 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
    v13 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_15;
  }
  if ( v2[23] >= 0 )
    goto LABEL_107;
  v8 = *(_DWORD *)(sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL) + 8);
  if ( v2[23] >= 0 )
    goto LABEL_108;
  v9 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v11 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v9 + v10 - 4) - v8);
  v12 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
  v13 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
LABEL_15:
  v20 = (__int64 *)(v13 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
  if ( &v2[v11] == (char *)v20 )
    goto LABEL_23;
  v21 = &v2[v11];
  do
  {
    while ( 1 )
    {
      v22 = *v20;
      if ( *(_BYTE *)(*(_QWORD *)*v20 + 8LL) == 15 )
      {
        v23 = *(_BYTE *)(v22 + 16);
        if ( v23 > 3u )
        {
          if ( v23 == 5 )
          {
            if ( (unsigned int)*(unsigned __int16 *)(v22 + 18) - 51 > 1 )
            {
              v79 &= 0xFFFFFFFF00000000LL;
              if ( (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), v22, v79, 0) )
                sub_1389140(a1, v22);
            }
          }
          else
          {
            v83 &= 0xFFFFFFFF00000000LL;
            sub_13848E0(*(_QWORD *)(a1 + 24), v22, v83, 0);
          }
          goto LABEL_17;
        }
        v80 = *(_QWORD *)(a1 + 24);
        v24 = sub_14C81A0(*v20);
        v87 &= 0xFFFFFFFF00000000LL;
        if ( (unsigned __int8)sub_13848E0(v80, v22, v87, v24) )
          break;
      }
LABEL_17:
      v20 += 3;
      if ( v21 == (char *)v20 )
        goto LABEL_22;
    }
    v20 += 3;
    v81 = *(_QWORD *)(a1 + 24);
    v25 = ((__int64 (*)(void))sub_14C8160)();
    v78 = v78 & 0xFFFFFFFF00000000LL | 1;
    sub_13848E0(v81, v22, 1u, v25);
  }
  while ( v21 != (char *)v20 );
LABEL_22:
  v3 = a1;
LABEL_23:
  if ( *(_BYTE *)(*(_QWORD *)v2 + 8LL) == 15 )
    sub_1389430(v3, (__int64)v2, 0);
  if ( !(unsigned __int8)sub_140B160(v2, *(_QWORD *)(v3 + 16), 0) )
  {
    v26 = sub_140B650(v2, *(_QWORD *)(v3 + 16));
    if ( !v26 )
    {
      v90 = v92;
      v91 = 0x400000000LL;
      v27 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v88[0] & 4) != 0 )
      {
        v28 = *(_QWORD *)(v27 - 24);
        if ( *(_BYTE *)(v28 + 16) )
        {
LABEL_40:
          v29 = 0xFFFFFFFFLL;
          v30 = (__int64 *)(v27 + 56);
          if ( (unsigned __int8)sub_1560260(v27 + 56, 0xFFFFFFFFLL, 36) )
            goto LABEL_41;
          if ( *(char *)(v27 + 23) >= 0 )
            goto LABEL_111;
          v39 = sub_1648A40(v27);
          v85 = v40 + v39;
          v41 = 0;
          if ( *(char *)(v27 + 23) < 0 )
            v41 = sub_1648A40(v27);
          if ( !(unsigned int)((v85 - v41) >> 4) )
          {
LABEL_111:
            v42 = *(_QWORD *)(v27 - 24);
            if ( !*(_BYTE *)(v42 + 16) )
            {
              v30 = &v89;
              v29 = 0xFFFFFFFFLL;
              v89 = *(_QWORD *)(v42 + 112);
              if ( (unsigned __int8)sub_1560260(&v89, 0xFFFFFFFFLL, 36) )
                goto LABEL_41;
            }
          }
          v29 = 0xFFFFFFFFLL;
          v30 = (__int64 *)(v27 + 56);
          if ( (unsigned __int8)sub_1560260(v27 + 56, 0xFFFFFFFFLL, 37) )
            goto LABEL_41;
          if ( *(char *)(v27 + 23) < 0 )
          {
            v43 = sub_1648A40(v27);
            v45 = v43 + v44;
            if ( *(char *)(v27 + 23) < 0 )
              v26 = sub_1648A40(v27);
            if ( v26 != v45 )
            {
              while ( *(_DWORD *)(*(_QWORD *)v26 + 8LL) <= 1u )
              {
                v26 += 16;
                if ( v45 == v26 )
                  goto LABEL_96;
              }
              goto LABEL_76;
            }
          }
LABEL_96:
          v38 = *(_QWORD *)(v27 - 24);
          if ( *(_BYTE *)(v38 + 16) )
          {
LABEL_76:
            v30 = v88;
            v46 = sub_1389B50(v88);
            v47 = (v88[0] & 0xFFFFFFFFFFFFFFF8LL)
                - 24LL * (*(_DWORD *)((v88[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
            if ( v46 != v47 )
            {
              v86 = v2;
              v48 = v46;
              do
              {
                v49 = *(_QWORD *)v47;
                if ( *(_BYTE *)(**(_QWORD **)v47 + 8LL) == 15 )
                {
                  v50 = *(_QWORD *)(v3 + 24);
                  v51 = sub_14C8190();
                  v52 = *(unsigned int *)(v50 + 24);
                  v53 = v51;
                  if ( !(_DWORD)v52 )
                    goto LABEL_106;
                  v54 = *(_QWORD *)(v50 + 8);
                  v55 = (v52 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
                  v56 = (__int64 *)(v54 + 32LL * v55);
                  v57 = *v56;
                  if ( v49 != *v56 )
                  {
                    v72 = 1;
                    while ( v57 != -8 )
                    {
                      v73 = v72 + 1;
                      v74 = ((_DWORD)v52 - 1) & (v55 + v72);
                      v55 = v74;
                      v56 = (__int64 *)(v54 + 32 * v74);
                      v57 = *v56;
                      if ( v49 == *v56 )
                        goto LABEL_82;
                      v72 = v73;
                    }
LABEL_106:
                    BUG();
                  }
LABEL_82:
                  if ( v56 == (__int64 *)(v54 + 32 * v52) )
                    goto LABEL_106;
                  v58 = v56[1];
                  if ( !(-1227133513 * (unsigned int)((v56[2] - v58) >> 3)) )
                    goto LABEL_106;
                  *(_QWORD *)(v58 + 48) |= v53;
                  v82 = *(_QWORD *)(v3 + 24);
                  v59 = ((__int64 (*)(void))sub_14C8160)();
                  v30 = (__int64 *)v82;
                  v29 = v49;
                  v77 = v77 & 0xFFFFFFFF00000000LL | 1;
                  sub_13848E0(v82, v49, 1u, v59);
                }
                v47 += 24LL;
              }
              while ( v48 != v47 );
              v2 = v86;
            }
            goto LABEL_41;
          }
          goto LABEL_62;
        }
      }
      else
      {
        v28 = *(_QWORD *)(v27 - 72);
        if ( *(_BYTE *)(v28 + 16) )
          goto LABEL_47;
      }
      v92[0] = v28;
      LODWORD(v91) = 1;
      if ( (unsigned __int8)sub_138E5C0(v3, v88[0], &v90) )
      {
LABEL_42:
        if ( v90 != v92 )
          _libc_free((unsigned __int64)v90);
        return;
      }
      v27 = v88[0] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v88[0] & 4) != 0 )
        goto LABEL_40;
LABEL_47:
      v29 = 0xFFFFFFFFLL;
      v30 = (__int64 *)(v27 + 56);
      if ( (unsigned __int8)sub_1560260(v27 + 56, 0xFFFFFFFFLL, 36) )
        goto LABEL_41;
      if ( *(char *)(v27 + 23) >= 0 )
        goto LABEL_112;
      v31 = sub_1648A40(v27);
      v84 = v32 + v31;
      v33 = 0;
      if ( *(char *)(v27 + 23) < 0 )
        v33 = sub_1648A40(v27);
      if ( !(unsigned int)((v84 - v33) >> 4) )
      {
LABEL_112:
        v34 = *(_QWORD *)(v27 - 72);
        if ( !*(_BYTE *)(v34 + 16) )
        {
          v30 = &v89;
          v29 = 0xFFFFFFFFLL;
          v89 = *(_QWORD *)(v34 + 112);
          if ( (unsigned __int8)sub_1560260(&v89, 0xFFFFFFFFLL, 36) )
            goto LABEL_41;
        }
      }
      v29 = 0xFFFFFFFFLL;
      v30 = (__int64 *)(v27 + 56);
      if ( (unsigned __int8)sub_1560260(v27 + 56, 0xFFFFFFFFLL, 37) )
        goto LABEL_41;
      if ( *(char *)(v27 + 23) < 0 )
      {
        v35 = sub_1648A40(v27);
        v37 = v35 + v36;
        if ( *(char *)(v27 + 23) < 0 )
          v26 = sub_1648A40(v27);
        if ( v26 != v37 )
        {
          while ( *(_DWORD *)(*(_QWORD *)v26 + 8LL) <= 1u )
          {
            v26 += 16;
            if ( v37 == v26 )
              goto LABEL_61;
          }
          goto LABEL_76;
        }
      }
LABEL_61:
      v38 = *(_QWORD *)(v27 - 72);
      if ( *(_BYTE *)(v38 + 16) )
        goto LABEL_76;
LABEL_62:
      v30 = &v89;
      v29 = 0xFFFFFFFFLL;
      v89 = *(_QWORD *)(v38 + 112);
      if ( !(unsigned __int8)sub_1560260(&v89, 0xFFFFFFFFLL, 37) )
        goto LABEL_76;
LABEL_41:
      if ( *(_BYTE *)(*(_QWORD *)v2 + 8LL) == 15 )
      {
        v60 = (__int64 *)((v88[0] & 0xFFFFFFFFFFFFFFF8LL) - 72);
        v61 = v88[0] & 4;
        if ( (v88[0] & 4) != 0 )
          v60 = (__int64 *)((v88[0] & 0xFFFFFFFFFFFFFFF8LL) - 24);
        v62 = *v60;
        if ( *(_BYTE *)(v62 + 16)
          || (v29 = 0, v30 = (__int64 *)(v62 + 112), !(unsigned __int8)sub_1560260(v62 + 112, 0, 20)) )
        {
          v63 = *(_QWORD *)(v3 + 24);
          v64 = sub_14C8160(v30, v29, v61);
          v65 = *(unsigned int *)(v63 + 24);
          v66 = v64;
          if ( !(_DWORD)v65 )
            goto LABEL_106;
          v67 = *(_QWORD *)(v63 + 8);
          v68 = (v65 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
          v69 = (char **)(v67 + 32LL * v68);
          v70 = *v69;
          if ( v2 != *v69 )
          {
            v75 = 1;
            while ( v70 != (char *)-8LL )
            {
              v76 = v75 + 1;
              v68 = (v65 - 1) & (v75 + v68);
              v69 = (char **)(v67 + 32LL * v68);
              v70 = *v69;
              if ( v2 == *v69 )
                goto LABEL_91;
              v75 = v76;
            }
            goto LABEL_106;
          }
LABEL_91:
          if ( v69 == (char **)(v67 + 32 * v65) )
            goto LABEL_106;
          v71 = v69[1];
          if ( !(-1227133513 * (unsigned int)((v69[2] - v71) >> 3)) )
            goto LABEL_106;
          *((_QWORD *)v71 + 6) |= v66;
        }
      }
      goto LABEL_42;
    }
  }
}
