// Function: sub_11DC340
// Address: 0x11dc340
//
__int64 __fastcall sub_11DC340(
        __int64 a1,
        unsigned __int8 *a2,
        unsigned __int8 *a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6)
{
  int v8; // esi
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int8 *v12; // rax
  unsigned __int8 *v14; // rdi
  unsigned __int8 *v15; // r8
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  _BYTE *v19; // r15
  __int64 **v20; // r13
  __int64 v21; // rax
  _BYTE *v22; // r14
  __int64 v23; // rdi
  __int64 (__fastcall *v24)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v25; // r15
  _QWORD *v26; // rax
  unsigned int *v27; // rbx
  __int64 v28; // r12
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 **v32; // rax
  _QWORD *v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r14
  __int64 v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // r9
  unsigned __int64 v40; // r15
  unsigned int *v41; // r14
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // rdi
  __int64 (__fastcall *v45)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v46; // r14
  _QWORD *v47; // rdi
  __int64 **v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rbx
  __int64 v52; // rax
  char v53; // al
  _QWORD *v54; // rax
  unsigned __int64 v55; // r15
  unsigned int *v56; // r13
  __int64 v57; // rbx
  __int64 v58; // rdx
  unsigned int v59; // esi
  __int64 v60; // rdi
  __int64 (__fastcall *v61)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v62; // r13
  __int64 v63; // rdi
  __int64 v64; // rax
  char v65; // r14
  _QWORD *v66; // rax
  unsigned int *v67; // r14
  __int64 v68; // r13
  __int64 v69; // rdx
  unsigned int v70; // esi
  __int64 v71; // rdi
  __int64 v72; // rax
  char v73; // r14
  __int64 v74; // r9
  unsigned int *v75; // r14
  __int64 v76; // rdx
  unsigned int v77; // esi
  _QWORD *v78; // rax
  unsigned int *v79; // r15
  __int64 v80; // rdx
  unsigned int v81; // esi
  _QWORD *v82; // rax
  unsigned int *v83; // rbx
  __int64 v84; // r15
  __int64 v85; // rdx
  unsigned int v86; // esi
  __int64 v87; // [rsp-8h] [rbp-F8h]
  char v88; // [rsp+4h] [rbp-ECh]
  unsigned int v89; // [rsp+8h] [rbp-E8h]
  unsigned __int8 v90; // [rsp+10h] [rbp-E0h]
  __int64 v91; // [rsp+18h] [rbp-D8h]
  __int64 **v92; // [rsp+18h] [rbp-D8h]
  _QWORD *v93; // [rsp+20h] [rbp-D0h]
  __int64 v95; // [rsp+28h] [rbp-C8h]
  __int64 **v96; // [rsp+28h] [rbp-C8h]
  __int64 v97; // [rsp+28h] [rbp-C8h]
  __int64 v98; // [rsp+28h] [rbp-C8h]
  _QWORD v99[4]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v100; // [rsp+50h] [rbp-A0h]
  _QWORD v101[4]; // [rsp+60h] [rbp-90h] BYREF
  char v102; // [rsp+80h] [rbp-70h]
  char v103; // [rsp+81h] [rbp-6Fh]
  _QWORD v104[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v105; // [rsp+B0h] [rbp-40h]

  if ( a4 )
  {
    v8 = a4;
    if ( a4 != 1 )
    {
      v10 = 8 * a4;
      v11 = *(_QWORD *)(a6 + 40);
      v12 = *(unsigned __int8 **)(a6 + 32);
      v14 = &v12[v11];
      if ( v11 >> 2 > 0 )
      {
        v15 = &v12[4 * (v11 >> 2)];
        while ( v10 != *v12 )
        {
          if ( v10 == v12[1] )
          {
            ++v12;
            goto LABEL_10;
          }
          if ( v10 == v12[2] )
          {
            v12 += 2;
            goto LABEL_10;
          }
          if ( v10 == v12[3] )
          {
            v12 += 3;
            goto LABEL_10;
          }
          v12 += 4;
          if ( v15 == v12 )
          {
            v11 = v14 - v12;
            goto LABEL_28;
          }
        }
        goto LABEL_10;
      }
LABEL_28:
      if ( v11 != 2 )
      {
        if ( v11 != 3 )
        {
          if ( v11 != 1 )
            return 0;
LABEL_59:
          v25 = 0;
          if ( v10 != *v12 )
            return v25;
LABEL_10:
          if ( v14 == v12 || !(unsigned __int8)sub_988330(a1) )
            return 0;
          v16 = (_QWORD *)sub_BD5C60(a1);
          v91 = sub_BCCE00(v16, 8 * v8);
          v90 = sub_AE5260(a6, v91);
          if ( *a2 > 0x15u )
          {
            if ( *a3 <= 0x15u )
            {
              v19 = (_BYTE *)sub_9718F0((__int64)a3, v91, (_BYTE *)a6);
LABEL_74:
              if ( (unsigned __int8)sub_F518D0(a2, 0, a6, a1, 0, 0) < v90 )
                return 0;
              if ( v19 )
              {
LABEL_76:
                v71 = *(_QWORD *)(a5 + 48);
                v103 = 1;
                v101[0] = "lhsv";
                v102 = 3;
                v72 = sub_AA4E30(v71);
                v73 = sub_AE5020(v72, v91);
                v105 = 257;
                v93 = sub_BD2C40(80, unk_3F10A14);
                if ( v93 )
                {
                  sub_B4D190((__int64)v93, v91, (__int64)a2, (__int64)v104, 0, v73, 0, 0);
                  v74 = v87;
                }
                (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a5 + 88)
                                                                                            + 16LL))(
                  *(_QWORD *)(a5 + 88),
                  v93,
                  v101,
                  *(_QWORD *)(a5 + 56),
                  *(_QWORD *)(a5 + 64),
                  v74);
                v75 = *(unsigned int **)a5;
                v97 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
                if ( *(_QWORD *)a5 != v97 )
                {
                  do
                  {
                    v76 = *((_QWORD *)v75 + 1);
                    v77 = *v75;
                    v75 += 4;
                    sub_B99FD0((__int64)v93, v77, v76);
                  }
                  while ( (unsigned int *)v97 != v75 );
                }
                if ( v19 )
                {
LABEL_16:
                  v100 = 257;
                  v103 = 1;
                  v20 = *(__int64 ***)(a1 + 8);
                  v101[0] = "memcmp";
                  v102 = 3;
                  v21 = sub_92B530((unsigned int **)a5, 0x21u, (__int64)v93, v19, (__int64)v99);
                  v22 = (_BYTE *)v21;
                  if ( v20 == *(__int64 ***)(v21 + 8) )
                    return v21;
                  v23 = *(_QWORD *)(a5 + 80);
                  v24 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v23 + 120LL);
                  if ( v24 == sub_920130 )
                  {
                    if ( *v22 > 0x15u )
                    {
LABEL_22:
                      v105 = 257;
                      v26 = sub_BD2C40(72, unk_3F10A14);
                      v25 = (__int64)v26;
                      if ( v26 )
                        sub_B515B0((__int64)v26, (__int64)v22, (__int64)v20, (__int64)v104, 0, 0);
                      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
                        *(_QWORD *)(a5 + 88),
                        v25,
                        v101,
                        *(_QWORD *)(a5 + 56),
                        *(_QWORD *)(a5 + 64));
                      v27 = *(unsigned int **)a5;
                      v28 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
                      while ( (unsigned int *)v28 != v27 )
                      {
                        v29 = *((_QWORD *)v27 + 1);
                        v30 = *v27;
                        v27 += 4;
                        sub_B99FD0(v25, v30, v29);
                      }
                      return v25;
                    }
                    if ( (unsigned __int8)sub_AC4810(0x27u) )
                      v25 = sub_ADAB70(39, (unsigned __int64)v22, v20, 0);
                    else
                      v25 = sub_AA93C0(0x27u, (unsigned __int64)v22, (__int64)v20);
                  }
                  else
                  {
                    v25 = v24(v23, 39u, v22, (__int64)v20);
                  }
                  if ( !v25 )
                    goto LABEL_22;
                  return v25;
                }
LABEL_65:
                v63 = *(_QWORD *)(a5 + 48);
                v103 = 1;
                v101[0] = "rhsv";
                v102 = 3;
                v64 = sub_AA4E30(v63);
                v65 = sub_AE5020(v64, v91);
                v105 = 257;
                v66 = sub_BD2C40(80, unk_3F10A14);
                v19 = v66;
                if ( v66 )
                  sub_B4D190((__int64)v66, v91, (__int64)a3, (__int64)v104, 0, v65, 0, 0);
                (*(void (__fastcall **)(_QWORD, _BYTE *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
                  *(_QWORD *)(a5 + 88),
                  v19,
                  v101,
                  *(_QWORD *)(a5 + 56),
                  *(_QWORD *)(a5 + 64));
                v67 = *(unsigned int **)a5;
                v68 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
                if ( *(_QWORD *)a5 != v68 )
                {
                  do
                  {
                    v69 = *((_QWORD *)v67 + 1);
                    v70 = *v67;
                    v67 += 4;
                    sub_B99FD0((__int64)v19, v70, v69);
                  }
                  while ( (unsigned int *)v68 != v67 );
                }
                goto LABEL_16;
              }
              if ( (unsigned __int8)sub_F518D0(a3, 0, a6, a1, 0, 0) < v90 )
                return 0;
LABEL_95:
              v19 = 0;
              goto LABEL_76;
            }
          }
          else
          {
            v17 = sub_9718F0((__int64)a2, v91, (_BYTE *)a6);
            v93 = (_QWORD *)v17;
            if ( *a3 <= 0x15u )
            {
              v18 = sub_9718F0((__int64)a3, v91, (_BYTE *)a6);
              v19 = (_BYTE *)v18;
              if ( v93 )
              {
                if ( v18 )
                  goto LABEL_16;
LABEL_64:
                if ( (unsigned __int8)sub_F518D0(a3, 0, a6, a1, 0, 0) >= v90 )
                  goto LABEL_65;
                return 0;
              }
              goto LABEL_74;
            }
            if ( v17 )
              goto LABEL_64;
          }
          if ( (unsigned __int8)sub_F518D0(a2, 0, a6, a1, 0, 0) < v90
            || (unsigned __int8)sub_F518D0(a3, 0, a6, a1, 0, 0) < v90 )
          {
            return 0;
          }
          goto LABEL_95;
        }
        if ( v10 == *v12 )
          goto LABEL_10;
        ++v12;
      }
      if ( v10 == *v12 )
        goto LABEL_10;
      ++v12;
      goto LABEL_59;
    }
    v99[0] = "lhsv";
    v32 = *(__int64 ***)(a1 + 8);
    v33 = *(_QWORD **)(a5 + 72);
    v100 = 259;
    v92 = v32;
    v34 = sub_BCB2B0(v33);
    v35 = *(_QWORD *)(a5 + 48);
    v103 = 1;
    v36 = v34;
    v102 = 3;
    v101[0] = "lhsc";
    v37 = sub_AA4E30(v35);
    v105 = 257;
    v89 = (unsigned __int8)sub_AE5020(v37, v36);
    v38 = sub_BD2C40(80, unk_3F10A14);
    v39 = v89;
    v40 = (unsigned __int64)v38;
    if ( v38 )
      sub_B4D190((__int64)v38, v36, (__int64)a2, (__int64)v104, 0, v89, 0, 0);
    (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a5 + 88) + 16LL))(
      *(_QWORD *)(a5 + 88),
      v40,
      v101,
      *(_QWORD *)(a5 + 56),
      *(_QWORD *)(a5 + 64),
      v39);
    v41 = *(unsigned int **)a5;
    v95 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
    if ( *(_QWORD *)a5 != v95 )
    {
      do
      {
        v42 = *((_QWORD *)v41 + 1);
        v43 = *v41;
        v41 += 4;
        sub_B99FD0(v40, v43, v42);
      }
      while ( (unsigned int *)v95 != v41 );
    }
    if ( v92 == *(__int64 ***)(v40 + 8) )
    {
      v46 = (_BYTE *)v40;
      goto LABEL_44;
    }
    v44 = *(_QWORD *)(a5 + 80);
    v45 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v44 + 120LL);
    if ( v45 == sub_920130 )
    {
      if ( *(_BYTE *)v40 > 0x15u )
      {
LABEL_82:
        v105 = 257;
        v78 = sub_BD2C40(72, unk_3F10A14);
        v46 = v78;
        if ( v78 )
          sub_B515B0((__int64)v78, v40, (__int64)v92, (__int64)v104, 0, 0);
        (*(void (__fastcall **)(_QWORD, _BYTE *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
          *(_QWORD *)(a5 + 88),
          v46,
          v99,
          *(_QWORD *)(a5 + 56),
          *(_QWORD *)(a5 + 64));
        v79 = *(unsigned int **)a5;
        v98 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
        if ( *(_QWORD *)a5 != v98 )
        {
          do
          {
            v80 = *((_QWORD *)v79 + 1);
            v81 = *v79;
            v79 += 4;
            sub_B99FD0((__int64)v46, v81, v80);
          }
          while ( (unsigned int *)v98 != v79 );
        }
LABEL_44:
        v47 = *(_QWORD **)(a5 + 72);
        v99[0] = "rhsv";
        v48 = *(__int64 ***)(a1 + 8);
        v100 = 259;
        v96 = v48;
        v49 = sub_BCB2B0(v47);
        v50 = *(_QWORD *)(a5 + 48);
        v103 = 1;
        v51 = v49;
        v102 = 3;
        v101[0] = "rhsc";
        v52 = sub_AA4E30(v50);
        v53 = sub_AE5020(v52, v51);
        v105 = 257;
        v88 = v53;
        v54 = sub_BD2C40(80, unk_3F10A14);
        v55 = (unsigned __int64)v54;
        if ( v54 )
          sub_B4D190((__int64)v54, v51, (__int64)a3, (__int64)v104, 0, v88, 0, 0);
        (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
          *(_QWORD *)(a5 + 88),
          v55,
          v101,
          *(_QWORD *)(a5 + 56),
          *(_QWORD *)(a5 + 64));
        v56 = *(unsigned int **)a5;
        v57 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
        if ( *(_QWORD *)a5 != v57 )
        {
          do
          {
            v58 = *((_QWORD *)v56 + 1);
            v59 = *v56;
            v56 += 4;
            sub_B99FD0(v55, v59, v58);
          }
          while ( (unsigned int *)v57 != v56 );
        }
        if ( v96 == *(__int64 ***)(v55 + 8) )
        {
          v62 = (_BYTE *)v55;
          goto LABEL_54;
        }
        v60 = *(_QWORD *)(a5 + 80);
        v61 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v60 + 120LL);
        if ( v61 == sub_920130 )
        {
          if ( *(_BYTE *)v55 > 0x15u )
            goto LABEL_87;
          if ( (unsigned __int8)sub_AC4810(0x27u) )
            v62 = (_BYTE *)sub_ADAB70(39, v55, v96, 0);
          else
            v62 = (_BYTE *)sub_AA93C0(0x27u, v55, (__int64)v96);
        }
        else
        {
          v62 = (_BYTE *)v61(v60, 39u, (_BYTE *)v55, (__int64)v96);
        }
        if ( v62 )
        {
LABEL_54:
          v104[0] = "chardiff";
          v105 = 259;
          return sub_929DE0((unsigned int **)a5, v46, v62, (__int64)v104, 0, 0);
        }
LABEL_87:
        v105 = 257;
        v82 = sub_BD2C40(72, unk_3F10A14);
        v62 = v82;
        if ( v82 )
          sub_B515B0((__int64)v82, v55, (__int64)v96, (__int64)v104, 0, 0);
        (*(void (__fastcall **)(_QWORD, _BYTE *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
          *(_QWORD *)(a5 + 88),
          v62,
          v99,
          *(_QWORD *)(a5 + 56),
          *(_QWORD *)(a5 + 64));
        v83 = *(unsigned int **)a5;
        v84 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
        if ( *(_QWORD *)a5 != v84 )
        {
          do
          {
            v85 = *((_QWORD *)v83 + 1);
            v86 = *v83;
            v83 += 4;
            sub_B99FD0((__int64)v62, v86, v85);
          }
          while ( (unsigned int *)v84 != v83 );
        }
        goto LABEL_54;
      }
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v46 = (_BYTE *)sub_ADAB70(39, v40, v92, 0);
      else
        v46 = (_BYTE *)sub_AA93C0(0x27u, v40, (__int64)v92);
    }
    else
    {
      v46 = (_BYTE *)v45(v44, 39u, (_BYTE *)v40, (__int64)v92);
    }
    if ( v46 )
      goto LABEL_44;
    goto LABEL_82;
  }
  return sub_AD6530(*(_QWORD *)(a1 + 8), (__int64)a2);
}
