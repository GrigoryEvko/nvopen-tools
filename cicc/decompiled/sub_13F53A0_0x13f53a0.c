// Function: sub_13F53A0
// Address: 0x13f53a0
//
void __fastcall sub_13F53A0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r14
  unsigned __int64 v3; // r13
  __int64 *v4; // rax
  __int64 v5; // r12
  __int64 v6; // r12
  const char *v7; // rax
  _QWORD *v8; // r12
  _BYTE *v9; // rax
  _BYTE *v10; // rax
  __int64 v11; // rbx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rax
  unsigned int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rdx
  const char **v18; // r15
  __int64 v19; // r14
  const char *v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rdx
  const char **v23; // r14
  unsigned int v24; // r12d
  __int64 v25; // rsi
  const char *v26; // rdx
  __int64 v27; // rdi
  const char *v28; // rax
  __int64 v29; // rax
  unsigned int v30; // eax
  unsigned int v31; // ebx
  __int64 v32; // rax
  __int64 v33; // rbx
  unsigned int v34; // eax
  __int64 v35; // rdi
  __int64 v36; // rsi
  __int64 v37; // rcx
  unsigned int v38; // r14d
  _BYTE *v39; // rax
  unsigned int v40; // r12d
  __int64 *v41; // rbx
  __int64 v42; // rsi
  __int64 v43; // r13
  __int64 v44; // r13
  __int64 v45; // rax
  __int64 v46; // rax
  int v47; // eax
  __int64 v48; // r13
  __int64 v49; // rax
  _QWORD *v50; // rax
  int v51; // eax
  unsigned int v52; // ebx
  __int64 v53; // rax
  unsigned int v54; // ebx
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rbx
  unsigned __int64 v58; // r12
  unsigned int v59; // r15d
  unsigned __int64 v60; // rdx
  __int64 v61; // r15
  __int64 v62; // rbx
  __int64 v63; // rax
  unsigned int v64; // ebx
  __int64 v65; // rax
  unsigned int v66; // ebx
  __int64 v67; // rax
  __int64 v68; // [rsp+18h] [rbp-128h]
  __int64 v69; // [rsp+20h] [rbp-120h]
  const char *v70; // [rsp+28h] [rbp-118h]
  __int64 v71; // [rsp+30h] [rbp-110h]
  __int64 v72; // [rsp+30h] [rbp-110h]
  unsigned __int64 v73; // [rsp+30h] [rbp-110h]
  __int64 v74; // [rsp+30h] [rbp-110h]
  __int64 v75; // [rsp+30h] [rbp-110h]
  unsigned __int64 v76; // [rsp+40h] [rbp-100h]
  __int64 v78; // [rsp+60h] [rbp-E0h]
  unsigned __int64 v79; // [rsp+68h] [rbp-D8h]
  unsigned __int64 v80; // [rsp+68h] [rbp-D8h]
  __int64 v81; // [rsp+70h] [rbp-D0h]
  unsigned __int64 v82; // [rsp+70h] [rbp-D0h]
  _QWORD v83[2]; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v84; // [rsp+88h] [rbp-B8h] BYREF
  __int64 v85; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v86; // [rsp+98h] [rbp-A8h]
  __int64 v87; // [rsp+A0h] [rbp-A0h]
  __int64 v88; // [rsp+A8h] [rbp-98h]
  __int64 v89; // [rsp+B0h] [rbp-90h]
  const char *v90; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v91; // [rsp+C8h] [rbp-78h]
  _BYTE *v92; // [rsp+D0h] [rbp-70h]
  __int64 v93; // [rsp+D8h] [rbp-68h]
  __int64 v94; // [rsp+E0h] [rbp-60h]
  _BYTE v95[88]; // [rsp+E8h] [rbp-58h] BYREF

  v2 = a1;
  v3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = (__int64 *)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  v83[0] = a2;
  if ( (a2 & 4) == 0 )
    v4 = (__int64 *)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 72);
  v5 = *v4;
  sub_13F46D0(a1, a2 & 0xFFFFFFFFFFFFFFF8LL, *v4, -1, 0, 0, 4u);
  v90 = 0;
  v91 = (__int64)v95;
  v92 = v95;
  v93 = 4;
  LODWORD(v94) = 0;
  v6 = sub_13F3D10(a1, v5, 0, (__int64)&v90);
  if ( v92 != (_BYTE *)v91 )
    _libc_free((unsigned __int64)v92);
  if ( *(_BYTE *)(v6 + 16) )
    goto LABEL_41;
  if ( ((*(_WORD *)(v6 + 18) >> 4) & 0x3FF) != ((*(unsigned __int16 *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 18) >> 2)
                                              & 0x3FFFDFFF) )
  {
    BYTE1(v92) = 1;
    v7 = "Undefined behavior: Caller and callee calling convention differ";
    goto LABEL_8;
  }
  v11 = *(_QWORD *)(v6 + 24);
  v12 = sub_1389B50(v83);
  v13 = v83[0] & 0xFFFFFFFFFFFFFFF8LL;
  v14 = 0xAAAAAAAAAAAAAAABLL
      * ((__int64)(v12
                 - ((v83[0] & 0xFFFFFFFFFFFFFFF8LL)
                  - 24LL * (*(_DWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3);
  v15 = *(_DWORD *)(v11 + 12) - 1;
  if ( *(_DWORD *)(v11 + 8) >> 8 )
  {
    if ( v15 <= (unsigned int)v14 )
      goto LABEL_17;
  }
  else if ( v15 == (_DWORD)v14 )
  {
LABEL_17:
    if ( **(_QWORD **)(v11 + 16) != *(_QWORD *)v3 )
    {
      BYTE1(v92) = 1;
      v28 = "Undefined behavior: Call return type mismatches callee return type";
LABEL_59:
      v8 = v2 + 30;
      v90 = v28;
      LOBYTE(v92) = 3;
      sub_16E2CE0(&v90, v2 + 30);
      v39 = (_BYTE *)v2[33];
      if ( (unsigned __int64)v39 >= v2[32] )
      {
        sub_16E7DE0(v2 + 30, 10);
      }
      else
      {
        v2[33] = v39 + 1;
        *v39 = 10;
      }
LABEL_11:
      if ( *(_BYTE *)(v3 + 16) <= 0x17u )
      {
        sub_15537D0(v3, v8, 1);
        v10 = (_BYTE *)v2[33];
        if ( (unsigned __int64)v10 < v2[32] )
          goto LABEL_13;
      }
      else
      {
        sub_155C2B0(v3, v8, 0);
        v10 = (_BYTE *)v2[33];
        if ( (unsigned __int64)v10 < v2[32] )
        {
LABEL_13:
          v2[33] = v10 + 1;
          *v10 = 10;
          return;
        }
      }
      sub_16E7DE0(v8, 10);
      return;
    }
    if ( (*(_BYTE *)(v6 + 18) & 1) != 0 )
    {
      sub_15E08E0(v6);
      v16 = *(_QWORD *)(v6 + 88);
      if ( (*(_BYTE *)(v6 + 18) & 1) != 0 )
        sub_15E08E0(v6);
      v17 = *(_QWORD *)(v6 + 88);
      v13 = v83[0] & 0xFFFFFFFFFFFFFFF8LL;
    }
    else
    {
      v16 = *(_QWORD *)(v6 + 88);
      v17 = v16;
    }
    v78 = v17 + 40LL * *(_QWORD *)(v6 + 96);
    v18 = (const char **)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
    v79 = sub_1389B50(v83);
    if ( (const char **)v79 != v18 )
    {
      v19 = v16;
      v76 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        if ( v78 != v19 )
        {
          v20 = *v18;
          v81 = v19 + 40;
          if ( *(_QWORD *)v19 != *(_QWORD *)*v18 )
          {
            BYTE1(v92) = 1;
            v3 = v76;
            v28 = "Undefined behavior: Call argument type mismatches callee parameter type";
            v2 = a1;
            goto LABEL_59;
          }
          if ( (unsigned __int8)sub_15E04B0(v19) )
          {
            if ( *(_BYTE *)(*(_QWORD *)v20 + 8LL) == 15 )
            {
              v21 = *(_DWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF;
              v84 = *(_QWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
              v22 = 24 * v21;
              if ( v79 != (v83[0] & 0xFFFFFFFFFFFFFFF8LL) - v22 )
              {
                v71 = v19;
                v23 = (const char **)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) - v22);
                v70 = v20;
                v24 = 0;
                do
                {
                  v25 = v24++;
                  if ( (unsigned __int8)sub_1560290(&v84, v25, 6) != 1
                    && v18 != v23
                    && *(_BYTE *)(*(_QWORD *)*v23 + 8LL) == 15 )
                  {
                    v26 = *v18;
                    v90 = *v23;
                    v91 = -1;
                    v85 = (__int64)v26;
                    v27 = a1[22];
                    v92 = 0;
                    v93 = 0;
                    v94 = 0;
                    v86 = -1;
                    v87 = 0;
                    v88 = 0;
                    v89 = 0;
                    if ( (unsigned __int8)(sub_134CB50(v27, (__int64)&v85, (__int64)&v90) - 2) <= 1u )
                    {
                      BYTE1(v92) = 1;
                      v3 = v76;
                      v28 = "Unusual: noalias argument aliases another argument";
                      v2 = a1;
                      goto LABEL_59;
                    }
                  }
                  v23 += 3;
                }
                while ( (const char **)v79 != v23 );
                v19 = v71;
                v20 = v70;
              }
            }
          }
          if ( (unsigned __int8)sub_15E04F0(v19) && *(_BYTE *)(*(_QWORD *)v20 + 8LL) == 15 )
          {
            v33 = *(_QWORD *)(*(_QWORD *)v19 + 24LL);
            v34 = sub_15A9FE0(a1[21], v33);
            v35 = a1[21];
            v36 = v33;
            v37 = 1;
            v38 = v34;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v36 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v46 = *(_QWORD *)(v36 + 32);
                  v36 = *(_QWORD *)(v36 + 24);
                  v37 *= v46;
                  continue;
                case 1:
                  v45 = 16;
                  break;
                case 2:
                  v45 = 32;
                  break;
                case 3:
                case 9:
                  v45 = 64;
                  break;
                case 4:
                  v45 = 80;
                  break;
                case 5:
                case 6:
                  v45 = 128;
                  break;
                case 7:
                  v75 = v37;
                  v51 = sub_15A9520(v35, 0);
                  v37 = v75;
                  v45 = (unsigned int)(8 * v51);
                  break;
                case 0xB:
                  v45 = *(_DWORD *)(v36 + 8) >> 8;
                  break;
                case 0xD:
                  v74 = v37;
                  v50 = (_QWORD *)sub_15A9930(v35, v36);
                  v37 = v74;
                  v45 = 8LL * *v50;
                  break;
                case 0xE:
                  v48 = *(_QWORD *)(v36 + 32);
                  v68 = v37;
                  v69 = *(_QWORD *)(v36 + 24);
                  v73 = (unsigned int)sub_15A9FE0(v35, v69);
                  v49 = sub_127FA20(v35, v69);
                  v37 = v68;
                  v45 = 8 * v48 * v73 * ((v73 + ((unsigned __int64)(v49 + 7) >> 3) - 1) / v73);
                  break;
                case 0xF:
                  v72 = v37;
                  v47 = sub_15A9520(v35, *(_DWORD *)(v36 + 8) >> 8);
                  v37 = v72;
                  v45 = (unsigned int)(8 * v47);
                  break;
              }
              break;
            }
            sub_13F46D0(a1, v76, (__int64)v20, (unsigned __int64)(v45 * v37 + 7) >> 3, v38, v33, 3u);
            v19 = v81;
          }
          else
          {
            v19 = v81;
          }
        }
        v18 += 3;
        if ( (const char **)v79 == v18 )
        {
          v3 = v76;
          v2 = a1;
          break;
        }
      }
    }
LABEL_41:
    if ( (v83[0] & 4) != 0 && (*(_WORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 18) & 3u) - 1 <= 1 )
    {
      v40 = 0;
      v85 = *(_QWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
      v82 = sub_1389B50(v83);
      v41 = (__int64 *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL)
                      - 24LL * (*(_DWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
      if ( (__int64 *)v82 != v41 )
      {
        v80 = v3;
        do
        {
          v42 = v40++;
          v43 = *v41;
          if ( !(unsigned __int8)sub_1560290(&v85, v42, 6) )
          {
            v90 = 0;
            v91 = (__int64)v95;
            v92 = v95;
            v93 = 4;
            LODWORD(v94) = 0;
            v44 = sub_13F3D10(v2, v43, 1u, (__int64)&v90);
            if ( v92 != (_BYTE *)v91 )
              _libc_free((unsigned __int64)v92);
            if ( *(_BYTE *)(v44 + 16) == 53 )
            {
              BYTE1(v92) = 1;
              v3 = v80;
              v7 = "Undefined behavior: Call with \"tail\" keyword references alloca";
              goto LABEL_8;
            }
          }
          v41 += 3;
        }
        while ( (__int64 *)v82 != v41 );
        v3 = v80;
      }
    }
    if ( *(_BYTE *)(v3 + 16) != 78 )
      return;
    v29 = *(_QWORD *)(v3 - 24);
    if ( *(_BYTE *)(v29 + 16) || (*(_BYTE *)(v29 + 33) & 0x20) == 0 || !v3 )
      return;
    v30 = *(_DWORD *)(v29 + 36);
    if ( v30 != 201 )
    {
      if ( v30 <= 0xC9 )
      {
        switch ( v30 )
        {
          case 0x87u:
            v64 = sub_15603A0(v3 + 56, 0);
            v65 = sub_1649C60(*(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)));
            sub_13F46D0(v2, v3, v65, -1, v64, 0, 2u);
            v66 = sub_15603A0(v3 + 56, 1);
            v67 = sub_1649C60(*(_QWORD *)(v3 + 24 * (1LL - (*(_DWORD *)(v3 + 20) & 0xFFFFFFF))));
            sub_13F46D0(v2, v3, v67, -1, v66, 0, 1u);
            break;
          case 0x89u:
            v31 = sub_15603A0(v3 + 56, 0);
            v32 = sub_1649C60(*(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)));
            sub_13F46D0(v2, v3, v32, -1, v31, 0, 2u);
            break;
          case 0x85u:
            v52 = sub_15603A0(v3 + 56, 0);
            v53 = sub_1649C60(*(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)));
            sub_13F46D0(v2, v3, v53, -1, v52, 0, 2u);
            v54 = sub_15603A0(v3 + 56, 1);
            v55 = sub_1649C60(*(_QWORD *)(v3 + 24 * (1LL - (*(_DWORD *)(v3 + 20) & 0xFFFFFFF))));
            sub_13F46D0(v2, v3, v55, -1, v54, 0, 1u);
            v56 = *(_QWORD *)(v3 + 24 * (2LL - (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)));
            v90 = 0;
            v93 = 4;
            v91 = (__int64)v95;
            v92 = v95;
            LODWORD(v94) = 0;
            v57 = sub_13F3D10(v2, v56, 0, (__int64)&v90);
            if ( v92 != (_BYTE *)v91 )
              _libc_free((unsigned __int64)v92);
            v58 = 0;
            if ( *(_BYTE *)(v57 + 16) == 13 )
            {
              v59 = *(_DWORD *)(v57 + 32);
              if ( v59 > 0x40 )
              {
                if ( v59 - (unsigned int)sub_16A57B0(v57 + 24) <= 0x20 )
                  v58 = **(_QWORD **)(v57 + 24);
              }
              else
              {
                v58 = *(_QWORD *)(v57 + 24);
                if ( v58 )
                {
                  _BitScanReverse64(&v60, v58);
                  if ( 64 - ((unsigned int)v60 ^ 0x3F) >= 0x21 )
                    v58 = 0;
                }
              }
            }
            v61 = v2[22];
            v62 = sub_1649C60(*(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)));
            v63 = sub_1649C60(*(_QWORD *)(v3 + 24 * (1LL - (*(_DWORD *)(v3 + 20) & 0xFFFFFFF))));
            v90 = (const char *)v62;
            v91 = v58;
            v92 = 0;
            v93 = 0;
            v94 = 0;
            v85 = v63;
            v86 = v58;
            v87 = 0;
            v88 = 0;
            v89 = 0;
            if ( (unsigned __int8)sub_134CB50(v61, (__int64)&v85, (__int64)&v90) == 3 )
            {
              BYTE1(v92) = 1;
              v28 = "Undefined behavior: memcpy source and destination overlap";
              goto LABEL_59;
            }
            break;
        }
        return;
      }
      if ( v30 != 213 )
      {
        if ( v30 != 214 )
        {
          if ( v30 == 212 )
          {
            sub_13F46D0(
              v2,
              v3,
              *(_QWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL)
                        - 24LL * (*(_DWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)),
              -1,
              0,
              0,
              2u);
            sub_13F46D0(
              v2,
              v3,
              *(_QWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL)
                        - 24LL * (*(_DWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)
                        + 24),
              -1,
              0,
              0,
              1u);
          }
          return;
        }
        if ( !(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 40) + 56LL) + 24LL) + 8LL) >> 8) )
        {
          BYTE1(v92) = 1;
          v28 = "Undefined behavior: va_start called in a non-varargs function";
          goto LABEL_59;
        }
      }
    }
    sub_13F46D0(
      v2,
      v3,
      *(_QWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL)
                - 24LL * (*(_DWORD *)((v83[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)),
      -1,
      0,
      0,
      3u);
    return;
  }
  BYTE1(v92) = 1;
  v7 = "Undefined behavior: Call argument count mismatches callee argument count";
LABEL_8:
  v8 = v2 + 30;
  v90 = v7;
  LOBYTE(v92) = 3;
  sub_16E2CE0(&v90, v2 + 30);
  v9 = (_BYTE *)v2[33];
  if ( (unsigned __int64)v9 >= v2[32] )
  {
    sub_16E7DE0(v2 + 30, 10);
  }
  else
  {
    v2[33] = v9 + 1;
    *v9 = 10;
  }
  if ( v3 )
    goto LABEL_11;
}
