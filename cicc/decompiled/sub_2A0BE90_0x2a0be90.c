// Function: sub_2A0BE90
// Address: 0x2a0be90
//
unsigned __int64 __fastcall sub_2A0BE90(__int64 *a1, _QWORD *a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rax
  char *p_src; // rsi
  _QWORD *v6; // rdx
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r13
  __int64 v10; // r10
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // r9
  __int64 v13; // r8
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r12
  _QWORD *v18; // rdx
  unsigned __int64 v19; // r12
  __int64 v20; // rax
  _QWORD *v21; // rsi
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // r15
  unsigned __int64 v24; // r10
  _QWORD *v25; // rbx
  char *v26; // r11
  char *v27; // rsi
  _QWORD *v28; // rdx
  _QWORD *v29; // rdx
  unsigned __int64 v30; // rsi
  __int64 v31; // rax
  unsigned __int64 v32; // rsi
  __int64 v33; // rdx
  unsigned __int64 v34; // rax
  _QWORD *v35; // rdx
  unsigned __int64 v37; // rbx
  unsigned __int64 v38; // r8
  unsigned __int64 v39; // rax
  __int64 v40; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v41; // [rsp+8h] [rbp-98h]
  unsigned __int64 v42; // [rsp+10h] [rbp-90h]
  unsigned __int64 v43; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+28h] [rbp-78h]
  __int64 src; // [rsp+30h] [rbp-70h] BYREF
  __int64 v46; // [rsp+38h] [rbp-68h]
  __int64 v47; // [rsp+40h] [rbp-60h]
  __int64 v48; // [rsp+48h] [rbp-58h]
  __int64 v49; // [rsp+50h] [rbp-50h]
  __int64 v50; // [rsp+58h] [rbp-48h]
  __int64 v51; // [rsp+60h] [rbp-40h]
  __int64 v52; // [rsp+68h] [rbp-38h]
  char v53[48]; // [rsp+70h] [rbp-30h] BYREF

  v3 = 0;
  v4 = *a1;
  if ( *a1 == *a2 )
    return sub_AC25F0(&src, v3, (__int64)sub_C64CA0);
  p_src = (char *)&src;
  while ( 1 )
  {
    p_src += 8;
    v6 = (_QWORD *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v4 & 4) != 0 )
      break;
    if ( p_src > v53 )
      goto LABEL_5;
    *((_QWORD *)p_src - 1) = v6[17];
    if ( !v6 )
      goto LABEL_27;
    v35 = v6 + 18;
    *a1 = (__int64)v35;
    v4 = (__int64)v35;
LABEL_22:
    if ( v35 == (_QWORD *)*a2 )
    {
      v3 = p_src - (char *)&src;
      return sub_AC25F0(&src, v3, (__int64)sub_C64CA0);
    }
  }
  if ( p_src <= v53 )
  {
    *((_QWORD *)p_src - 1) = *(_QWORD *)(*v6 + 136LL);
LABEL_27:
    v4 = (unsigned __int64)(v6 + 1) | 4;
    *a1 = v4;
    v35 = (_QWORD *)v4;
    goto LABEL_22;
  }
LABEL_5:
  v7 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v8 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v43 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v8
           ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
           ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v8
          ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v46 + v7, 27));
  v9 = v50
     + v7
     - 0x4B6D499041670D8DLL * __ROL8__((char *)sub_C64CA0 + v51 + 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0, 22);
  v10 = src - 0x6D8ED9027DD26057LL * (_QWORD)sub_C64CA0;
  v11 = 0xB492B66FBE98F273LL
      * __ROL8__(
          0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
            ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
            ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
           ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
           ^ 0xB492B66FBE98F273LL)))
        + v8,
          31);
  v12 = 0x24AD9BEFA63C9CC0LL;
  v13 = v48 + v10 + v47 + v46;
  v44 = __ROR8__(v43 + v48 + v10 + v8, 21) + __ROL8__(v10 + v47 + v46, 20) + v10;
  v14 = v49
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (v8
          ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v8
         ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))))
      + v11;
  v15 = v14 + v50 + v51;
  v16 = v52 + v15;
  v17 = __ROL8__(v15, 20) + __ROR8__(v14 + v9 + v47 + v52, 21);
  v18 = (_QWORD *)*a2;
  v19 = v14 + v17;
  v20 = *a1;
  if ( *a2 == *a1 )
    goto LABEL_29;
  v21 = a2;
  v22 = v19;
  v23 = 64;
  v24 = v11;
  v25 = v21;
  while ( 2 )
  {
    v26 = (char *)&src;
    v27 = (char *)&src;
    if ( v18 == (_QWORD *)v20 )
      goto LABEL_15;
    while ( 2 )
    {
      while ( 1 )
      {
        v27 = v26;
        v26 += 8;
        v29 = (_QWORD *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v20 & 4) != 0 )
          break;
        if ( v26 > v53 )
          goto LABEL_14;
        *((_QWORD *)v26 - 1) = v29[17];
        if ( !v29 )
          goto LABEL_18;
        v28 = v29 + 18;
        *a1 = (__int64)v28;
        v20 = (__int64)v28;
        if ( v28 == (_QWORD *)*v25 )
          goto LABEL_19;
      }
      if ( v26 > v53 )
      {
LABEL_14:
        v23 += v27 - (char *)&src;
        goto LABEL_15;
      }
      *((_QWORD *)v26 - 1) = *(_QWORD *)(*v29 + 136LL);
LABEL_18:
      v20 = (unsigned __int64)(v29 + 1) | 4;
      *a1 = v20;
      if ( v20 != *v25 )
        continue;
      break;
    }
LABEL_19:
    v27 = v26;
    v23 += v26 - (char *)&src;
LABEL_15:
    v40 = v13;
    v41 = v22;
    v42 = v24;
    sub_2A0B400((char *)&src, v27, v53);
    v30 = v9 + v42;
    v31 = src - 0x4B6D499041670D8DLL * v44;
    v9 = v50 + v40 - 0x4B6D499041670D8DLL * __ROL8__(v51 + v44 + v9, 22);
    v24 = 0xB492B66FBE98F273LL * __ROL8__(v16 + v43, 31);
    v32 = v41 ^ (0xB492B66FBE98F273LL * __ROL8__(v40 + v46 + v30, 27));
    v33 = v31 + v47 + v46;
    v13 = v48 + v33;
    v44 = __ROR8__(v32 + v31 + v48 + v16, 21) + __ROL8__(v33, 20) + v31;
    v34 = v49 + v41 + v24;
    v16 = v52 + v34 + v50 + v51;
    v18 = (_QWORD *)*v25;
    v22 = __ROL8__(v34 + v50 + v51, 20) + v34 + __ROR8__(v9 + v34 + v47 + v52, 21);
    v20 = *a1;
    if ( *a1 != *v25 )
    {
      v43 = v32;
      continue;
    }
    break;
  }
  v11 = v24;
  v43 = v32;
  v19 = v22;
  v12 = 0xB492B66FBE98F273LL * ((v23 >> 47) ^ v23);
LABEL_29:
  v37 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v19 ^ v44)) >> 47) ^ v19 ^ (0x9DDFEA08EB382D69LL * (v19 ^ v44)))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v19 ^ v44)) >> 47) ^ v19 ^ (0x9DDFEA08EB382D69LL * (v19 ^ v44)))))
      + v12
      + v11;
  v38 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v16 ^ v13)) >> 47) ^ v16 ^ (0x9DDFEA08EB382D69LL * (v16 ^ v13)));
  v39 = 0x9DDFEA08EB382D69LL
      * (v37 ^ (0xB492B66FBE98F273LL * ((v9 >> 47) ^ v9) + v43 - 0x622015F714C7D297LL * ((v38 >> 47) ^ v38)));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v39 ^ v37 ^ (v39 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v39 ^ v37 ^ (v39 >> 47))));
}
