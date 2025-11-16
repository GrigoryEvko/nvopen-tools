// Function: sub_1019820
// Address: 0x1019820
//
unsigned __int8 *__fastcall sub_1019820(unsigned int a1, __int64 a2, unsigned __int8 *a3, __m128i *a4, unsigned int a5)
{
  __int64 **v9; // r15
  unsigned __int8 *result; // rax
  __int64 v11; // rsi
  unsigned int v12; // edx
  int v13; // eax
  bool v14; // al
  unsigned __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rdi
  unsigned int v18; // eax
  unsigned __int8 v19; // cl
  __int64 v20; // rdx
  unsigned __int8 *v21; // rax
  __int64 v22; // r8
  unsigned int v23; // eax
  unsigned __int64 v24; // rdx
  unsigned int v25; // r15d
  __int64 v26; // rax
  __int64 v27; // r8
  unsigned int v28; // esi
  unsigned __int64 v29; // rax
  bool v30; // zf
  unsigned int v31; // eax
  unsigned __int64 v32; // rcx
  unsigned int v33; // eax
  _BYTE *v34; // r15
  _BYTE *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // rdx
  unsigned __int64 *v40; // r15
  unsigned int v41; // r10d
  unsigned int v42; // edx
  unsigned __int64 v43; // rcx
  int v44; // r15d
  bool v45; // al
  unsigned __int8 *v46; // rsi
  _BYTE *v47; // rax
  __int64 v48; // r8
  unsigned int v49; // eax
  unsigned __int64 v50; // r15
  unsigned int v51; // ecx
  __int64 v52; // rax
  __int64 v53; // r8
  unsigned int v54; // esi
  unsigned __int64 v55; // rax
  unsigned int v56; // eax
  unsigned __int64 v57; // rcx
  unsigned int v58; // eax
  _BYTE *v59; // r15
  __int64 v60; // rdx
  _BYTE *v61; // rax
  _BYTE *v62; // rax
  int v63; // eax
  unsigned int v64; // eax
  unsigned __int64 v65; // rcx
  unsigned __int64 v66; // rcx
  __int64 v67; // rdx
  unsigned int v68; // eax
  unsigned __int64 v69; // rcx
  int v70; // eax
  unsigned __int64 v71; // rcx
  __int64 v72; // [rsp+0h] [rbp-C0h]
  unsigned int v73; // [rsp+0h] [rbp-C0h]
  __int64 v74; // [rsp+0h] [rbp-C0h]
  _BYTE *v75; // [rsp+8h] [rbp-B8h]
  __int64 v76; // [rsp+8h] [rbp-B8h]
  _BYTE *v77; // [rsp+8h] [rbp-B8h]
  __int64 v78; // [rsp+8h] [rbp-B8h]
  __int64 v79; // [rsp+8h] [rbp-B8h]
  __int64 v80; // [rsp+8h] [rbp-B8h]
  __int64 v81; // [rsp+8h] [rbp-B8h]
  __int64 v82; // [rsp+10h] [rbp-B0h]
  unsigned int v83; // [rsp+10h] [rbp-B0h]
  unsigned int v84; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v85; // [rsp+18h] [rbp-A8h]
  __int64 v86; // [rsp+18h] [rbp-A8h]
  __int64 v87; // [rsp+18h] [rbp-A8h]
  __int64 v88; // [rsp+18h] [rbp-A8h]
  __int64 v89; // [rsp+18h] [rbp-A8h]
  unsigned int v90; // [rsp+20h] [rbp-A0h]
  unsigned int v91; // [rsp+20h] [rbp-A0h]
  unsigned int v92; // [rsp+24h] [rbp-9Ch]
  unsigned __int8 *v93; // [rsp+28h] [rbp-98h]
  unsigned __int8 *v94; // [rsp+28h] [rbp-98h]
  unsigned __int64 v95; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v96; // [rsp+38h] [rbp-88h]
  unsigned __int64 v97; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v98; // [rsp+48h] [rbp-78h]
  __int64 v99; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v100; // [rsp+58h] [rbp-68h]
  __int64 v101; // [rsp+60h] [rbp-60h]
  unsigned int v102; // [rsp+68h] [rbp-58h]
  unsigned __int64 v103; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v104; // [rsp+78h] [rbp-48h]
  __int64 v105; // [rsp+80h] [rbp-40h]
  unsigned int v106; // [rsp+88h] [rbp-38h]

  v9 = *(__int64 ***)(a2 + 8);
  if ( (unsigned __int8)sub_1003090((__int64)a4, a3) || *a3 == 13 || (unsigned __int8)sub_FFFE90((__int64)a3) )
    return (unsigned __int8 *)sub_ACADE0(v9);
  if ( *(_BYTE *)a2 == 13 )
    return (unsigned __int8 *)a2;
  if ( (unsigned __int8)sub_1003090((__int64)a4, (unsigned __int8 *)a2) )
    goto LABEL_27;
  if ( (unsigned __int8)sub_FFFE90(a2) )
  {
    v17 = *(_QWORD *)(a2 + 8);
    return (unsigned __int8 *)sub_AD6530(v17, a2);
  }
  v92 = a1 - 19;
  if ( (unsigned __int8 *)a2 == a3 )
  {
    if ( a1 - 19 <= 1 )
      return (unsigned __int8 *)sub_AD64C0((__int64)v9, 1, 0);
LABEL_27:
    v17 = (__int64)v9;
    return (unsigned __int8 *)sub_AD6530(v17, a2);
  }
  v11 = (__int64)a3;
  sub_9AC330((__int64)&v99, (__int64)a3, 0, a4);
  v12 = v100;
  if ( !v100
    || (v100 <= 0x40
      ? (v14 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v100) == v99)
      : (v90 = v100, v13 = sub_C445E0((__int64)&v99), v12 = v90, v14 = v90 == v13),
        v14) )
  {
    result = (unsigned __int8 *)sub_ACADE0(v9);
    goto LABEL_19;
  }
  if ( v12 > 0x40 )
  {
    v84 = v12;
    v16 = sub_C44500((__int64)&v99);
    v12 = v84;
  }
  else
  {
    if ( v99 << (64 - (unsigned __int8)v12) == -1 )
      goto LABEL_33;
    _BitScanReverse64(&v15, ~(v99 << (64 - (unsigned __int8)v12)));
    v16 = v15 ^ 0x3F;
  }
  if ( --v12 != v16 )
  {
LABEL_33:
    v18 = a1;
    v19 = *(_BYTE *)a2;
    LOBYTE(v12) = a1 == 20;
    LOBYTE(v18) = a1 == 23;
    v20 = v18 | v12;
    if ( *(_BYTE *)a2 != 46 )
    {
      if ( !a5 )
        goto LABEL_72;
      v91 = a5 - 1;
      if ( (_BYTE)v20 )
      {
        if ( v19 == 52 )
        {
          v21 = *(unsigned __int8 **)(a2 - 32);
          if ( !v21 || a3 != v21 )
          {
            v82 = *(_QWORD *)(a2 + 8);
LABEL_40:
            v22 = (__int64)(a3 + 24);
            if ( *a3 != 17 )
            {
              v60 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a3 + 1) + 8LL) - 17;
              if ( (unsigned int)v60 > 1 )
                goto LABEL_69;
              if ( *a3 > 0x15u )
                goto LABEL_69;
              v61 = sub_AD7630((__int64)a3, 0, v60);
              if ( !v61 || *v61 != 17 )
                goto LABEL_69;
              v22 = (__int64)(v61 + 24);
            }
            v23 = *(_DWORD *)(v22 + 8);
            v24 = *(_QWORD *)v22;
            v25 = v23 - 1;
            if ( v23 <= 0x40 )
            {
              if ( v24 != 1LL << v25 )
              {
                if ( ((1LL << v25) & v24) == 0 )
                {
LABEL_43:
                  v98 = *(_DWORD *)(v22 + 8);
                  if ( v98 > 0x40 )
                  {
                    v81 = v22;
                    sub_C43780((__int64)&v97, (const void **)v22);
                    v22 = v81;
                  }
                  else
                  {
                    v97 = *(_QWORD *)v22;
                  }
LABEL_45:
                  v72 = v22;
                  v26 = sub_AD8D80(v82, (__int64)&v97);
                  v27 = v72;
                  v75 = (_BYTE *)v26;
                  if ( v98 > 0x40 && v97 )
                  {
                    j_j___libc_free_0_0(v97);
                    v27 = v72;
                  }
                  v28 = *(_DWORD *)(v27 + 8);
                  if ( v28 > 0x40 )
                    v29 = *(_QWORD *)(*(_QWORD *)v27 + 8LL * ((v28 - 1) >> 6));
                  else
                    v29 = *(_QWORD *)v27;
                  v30 = (v29 & (1LL << ((unsigned __int8)v28 - 1))) == 0;
                  v31 = *(_DWORD *)(v27 + 8);
                  if ( v30 )
                  {
                    v98 = *(_DWORD *)(v27 + 8);
                    if ( v31 <= 0x40 )
                    {
                      v97 = *(_QWORD *)v27;
                      goto LABEL_183;
                    }
                    sub_C43780((__int64)&v97, (const void **)v27);
LABEL_57:
                    v31 = v98;
                    if ( v98 > 0x40 )
                    {
                      sub_C43D10((__int64)&v97);
                      goto LABEL_59;
                    }
LABEL_183:
                    v66 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v31;
                    if ( !v31 )
                      v66 = 0;
                    v97 = v66 & ~v97;
LABEL_59:
                    sub_C46250((__int64)&v97);
                    v33 = v98;
                    v98 = 0;
                    v104 = v33;
                    v103 = v97;
                    v34 = (_BYTE *)sub_AD8D80(v82, (__int64)&v103);
                    if ( v104 > 0x40 && v103 )
                      j_j___libc_free_0_0(v103);
                    if ( v98 > 0x40 && v97 )
                      j_j___libc_free_0_0(v97);
                    v35 = (_BYTE *)sub_1012FB0(0x26u, (_BYTE *)a2, v34, a4->m128i_i64, v91);
                    if ( !v35 )
                      goto LABEL_69;
                    if ( *v35 > 0x15u )
                      goto LABEL_69;
                    if ( !sub_AD7930(v35, a2, v36, v37, v38) )
                      goto LABEL_69;
                    v11 = a2;
                    if ( !sub_1016CD0(0x28u, (_BYTE *)a2, v75, a4->m128i_i64, v91) )
                      goto LABEL_69;
                    goto LABEL_97;
                  }
                  v104 = *(_DWORD *)(v27 + 8);
                  if ( v31 > 0x40 )
                  {
                    sub_C43780((__int64)&v103, (const void **)v27);
                    v31 = v104;
                    if ( v104 > 0x40 )
                    {
                      sub_C43D10((__int64)&v103);
LABEL_56:
                      sub_C46250((__int64)&v103);
                      v98 = v104;
                      v97 = v103;
                      goto LABEL_57;
                    }
                  }
                  else
                  {
                    v103 = *(_QWORD *)v27;
                  }
                  v32 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v31;
                  if ( !v31 )
                    v32 = 0;
                  v103 = v32 & ~v103;
                  goto LABEL_56;
                }
LABEL_173:
                v64 = *(_DWORD *)(v22 + 8);
                v104 = v64;
                if ( v64 > 0x40 )
                {
                  v88 = v22;
                  sub_C43780((__int64)&v103, (const void **)v22);
                  v64 = v104;
                  v22 = v88;
                  if ( v104 > 0x40 )
                  {
                    sub_C43D10((__int64)&v103);
                    v22 = v88;
LABEL_178:
                    v78 = v22;
                    sub_C46250((__int64)&v103);
                    v22 = v78;
                    v98 = v104;
                    v97 = v103;
                    goto LABEL_45;
                  }
                }
                else
                {
                  v103 = *(_QWORD *)v22;
                }
                v65 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v64;
                if ( !v64 )
                  v65 = 0;
                v103 = v65 & ~v103;
                goto LABEL_178;
              }
            }
            else
            {
              if ( (*(_QWORD *)(v24 + 8LL * (v25 >> 6)) & (1LL << v25)) == 0 )
                goto LABEL_43;
              v86 = v22;
              v63 = sub_C44590(v22);
              v22 = v86;
              if ( v63 != v25 )
                goto LABEL_173;
            }
            v11 = a2;
            v45 = sub_1016CD0(0x21u, (_BYTE *)a2, a3, a4->m128i_i64, v91);
LABEL_96:
            if ( v45 )
              goto LABEL_97;
LABEL_69:
            if ( a5 == 3 )
            {
              result = (unsigned __int8 *)sub_FFE7D0(a1, a2, a3, a4->m128i_i64[0], a4[2].m128i_i64[1]);
              if ( result )
                goto LABEL_19;
            }
            v19 = *(_BYTE *)a2;
LABEL_72:
            if ( v19 != 86 && *a3 != 86 )
              goto LABEL_74;
            goto LABEL_73;
          }
LABEL_97:
          if ( v92 > 1 )
            goto LABEL_18;
          goto LABEL_98;
        }
        v82 = *(_QWORD *)(a2 + 8);
        if ( v19 == 17 )
        {
          v48 = a2 + 24;
          goto LABEL_110;
        }
LABEL_105:
        if ( (unsigned int)*(unsigned __int8 *)(v82 + 8) - 17 > 1 )
          goto LABEL_40;
        if ( v19 > 0x15u )
          goto LABEL_40;
        v47 = sub_AD7630(a2, 0, v20);
        if ( !v47 || *v47 != 17 )
          goto LABEL_40;
        v48 = (__int64)(v47 + 24);
LABEL_110:
        v49 = *(_DWORD *)(v48 + 8);
        v50 = *(_QWORD *)v48;
        v51 = v49 - 1;
        if ( v49 <= 0x40 )
        {
          if ( 1LL << v51 == v50 )
            goto LABEL_40;
          v67 = 1LL << v51;
        }
        else
        {
          v73 = v49 - 1;
          v76 = 1LL << v51;
          v50 = *(_QWORD *)(v50 + 8LL * (v51 >> 6));
          if ( (v50 & (1LL << v51)) == 0 )
            goto LABEL_112;
          v87 = v48;
          v70 = sub_C44590(v48);
          v48 = v87;
          v67 = v76;
          if ( v73 == v70 )
            goto LABEL_40;
        }
        if ( (v67 & v50) == 0 )
        {
LABEL_112:
          v98 = *(_DWORD *)(v48 + 8);
          if ( v98 > 0x40 )
          {
            v80 = v48;
            sub_C43780((__int64)&v97, (const void **)v48);
            v48 = v80;
          }
          else
          {
            v97 = *(_QWORD *)v48;
          }
LABEL_114:
          v74 = v48;
          v52 = sub_AD8D80(v82, (__int64)&v97);
          v53 = v74;
          v77 = (_BYTE *)v52;
          if ( v98 > 0x40 && v97 )
          {
            j_j___libc_free_0_0(v97);
            v53 = v74;
          }
          v54 = *(_DWORD *)(v53 + 8);
          if ( v54 > 0x40 )
            v55 = *(_QWORD *)(*(_QWORD *)v53 + 8LL * ((v54 - 1) >> 6));
          else
            v55 = *(_QWORD *)v53;
          v30 = (v55 & (1LL << ((unsigned __int8)v54 - 1))) == 0;
          v56 = *(_DWORD *)(v53 + 8);
          if ( v30 )
          {
            v98 = *(_DWORD *)(v53 + 8);
            if ( v56 <= 0x40 )
            {
              v97 = *(_QWORD *)v53;
              goto LABEL_206;
            }
            sub_C43780((__int64)&v97, (const void **)v53);
LABEL_126:
            v56 = v98;
            if ( v98 > 0x40 )
            {
              sub_C43D10((__int64)&v97);
              goto LABEL_128;
            }
LABEL_206:
            v71 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v56;
            if ( !v56 )
              v71 = 0;
            v97 = v71 & ~v97;
LABEL_128:
            sub_C46250((__int64)&v97);
            v58 = v98;
            v98 = 0;
            v104 = v58;
            v103 = v97;
            v59 = (_BYTE *)sub_AD8D80(v82, (__int64)&v103);
            if ( v104 > 0x40 && v103 )
              j_j___libc_free_0_0(v103);
            if ( v98 > 0x40 && v97 )
              j_j___libc_free_0_0(v97);
            v11 = (__int64)a3;
            if ( !sub_1016CD0(0x28u, a3, v59, a4->m128i_i64, v91) )
            {
              v11 = (__int64)a3;
              if ( !sub_1016CD0(0x26u, a3, v77, a4->m128i_i64, v91) )
                goto LABEL_40;
            }
            goto LABEL_97;
          }
          v104 = *(_DWORD *)(v53 + 8);
          if ( v56 > 0x40 )
          {
            sub_C43780((__int64)&v103, (const void **)v53);
            v56 = v104;
            if ( v104 > 0x40 )
            {
              sub_C43D10((__int64)&v103);
LABEL_125:
              sub_C46250((__int64)&v103);
              v98 = v104;
              v97 = v103;
              goto LABEL_126;
            }
          }
          else
          {
            v103 = *(_QWORD *)v53;
          }
          v57 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v56;
          if ( !v56 )
            v57 = 0;
          v103 = v57 & ~v103;
          goto LABEL_125;
        }
        v68 = *(_DWORD *)(v48 + 8);
        v104 = v68;
        if ( v68 > 0x40 )
        {
          v89 = v48;
          sub_C43780((__int64)&v103, (const void **)v48);
          v68 = v104;
          v48 = v89;
          if ( v104 > 0x40 )
          {
            sub_C43D10((__int64)&v103);
            v48 = v89;
LABEL_195:
            v79 = v48;
            sub_C46250((__int64)&v103);
            v48 = v79;
            v98 = v104;
            v97 = v103;
            goto LABEL_114;
          }
        }
        else
        {
          v103 = *(_QWORD *)v48;
        }
        v69 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v68;
        if ( !v68 )
          v69 = 0;
        v103 = v69 & ~v103;
        goto LABEL_195;
      }
      goto LABEL_79;
    }
    result = *(unsigned __int8 **)(a2 - 64);
    v46 = *(unsigned __int8 **)(a2 - 32);
    if ( result )
    {
      if ( a3 == v46 )
      {
        if ( !v46 )
          goto LABEL_102;
      }
      else
      {
        if ( !v46 || a3 != result )
          goto LABEL_102;
        result = *(unsigned __int8 **)(a2 - 32);
      }
      v11 = a4[4].m128i_u8[0];
      if ( (_BYTE)v20 )
      {
        if ( (_BYTE)v11 && (*(_BYTE *)(a2 + 1) & 4) != 0
          || *result == 49 && (v11 = *((_QWORD *)result - 4)) != 0 && a3 == (unsigned __int8 *)v11 )
        {
LABEL_143:
          if ( v92 <= 1 )
            goto LABEL_19;
LABEL_98:
          result = (unsigned __int8 *)sub_AD6530(*(_QWORD *)(a2 + 8), v11);
          goto LABEL_19;
        }
      }
      else
      {
        if ( (_BYTE)v11 && (*(_BYTE *)(a2 + 1) & 2) != 0 )
          goto LABEL_143;
        if ( *result == 48 )
        {
          v11 = *((_QWORD *)result - 4);
          if ( a3 == (unsigned __int8 *)v11 )
          {
            if ( v11 )
              goto LABEL_143;
          }
        }
      }
    }
LABEL_102:
    if ( !a5 )
    {
      if ( *a3 != 86 )
      {
LABEL_156:
        result = 0;
        if ( *a3 != 84 )
          goto LABEL_19;
LABEL_75:
        result = (unsigned __int8 *)sub_101CAB0(a1, a2, a3, a4, a5);
        goto LABEL_19;
      }
LABEL_73:
      result = (unsigned __int8 *)sub_101C8A0(a1, a2, a3, a4, a5);
      if ( result )
        goto LABEL_19;
LABEL_74:
      if ( *(_BYTE *)a2 == 84 )
        goto LABEL_75;
      goto LABEL_156;
    }
    v91 = a5 - 1;
    if ( (_BYTE)v20 )
    {
      v82 = *(_QWORD *)(a2 + 8);
      goto LABEL_105;
    }
LABEL_79:
    v39 = *a3;
    v40 = (unsigned __int64 *)(a3 + 24);
    if ( (_BYTE)v39 != 17 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a3 + 1) + 8LL) - 17 > 1 )
        goto LABEL_95;
      if ( (unsigned __int8)v39 > 0x15u )
        goto LABEL_95;
      v62 = sub_AD7630((__int64)a3, 0, v39);
      if ( !v62 || *v62 != 17 )
        goto LABEL_95;
      v40 = (unsigned __int64 *)(v62 + 24);
    }
    sub_9AC330((__int64)&v103, a2, 0, a4);
    v41 = v104;
    v98 = v104;
    if ( v104 > 0x40 )
    {
      sub_C43780((__int64)&v97, (const void **)&v103);
      v42 = v98;
      v41 = v98;
      if ( v98 > 0x40 )
      {
        sub_C43D10((__int64)&v97);
        v42 = v98;
        goto LABEL_85;
      }
    }
    else
    {
      v42 = v104;
      v97 = v103;
    }
    v43 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v41;
    if ( !v41 )
      v43 = 0;
    v97 = v43 & ~v97;
LABEL_85:
    v11 = (__int64)v40;
    v96 = v42;
    v83 = v42;
    v95 = v97;
    v85 = v97;
    v44 = sub_C49970((__int64)&v95, v40);
    if ( v83 > 0x40 && v85 )
      j_j___libc_free_0_0(v85);
    if ( v106 > 0x40 && v105 )
      j_j___libc_free_0_0(v105);
    if ( v104 > 0x40 && v103 )
      j_j___libc_free_0_0(v103);
    if ( v44 < 0 )
      goto LABEL_97;
LABEL_95:
    v11 = a2;
    v45 = sub_1016CD0(0x24u, (_BYTE *)a2, a3, a4->m128i_i64, v91);
    goto LABEL_96;
  }
  if ( v92 <= 1 )
  {
LABEL_18:
    result = (unsigned __int8 *)a2;
    goto LABEL_19;
  }
  result = (unsigned __int8 *)sub_AD6530((__int64)v9, (__int64)a3);
LABEL_19:
  if ( v102 > 0x40 && v101 )
  {
    v93 = result;
    j_j___libc_free_0_0(v101);
    result = v93;
  }
  if ( v100 > 0x40 )
  {
    if ( v99 )
    {
      v94 = result;
      j_j___libc_free_0_0(v99);
      return v94;
    }
  }
  return result;
}
