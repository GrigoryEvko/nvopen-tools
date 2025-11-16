// Function: sub_F58E90
// Address: 0xf58e90
//
unsigned __int64 __fastcall sub_F58E90(__int64 *a1, __int64 *a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // rsi
  char *v5; // r10
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  __int64 v9; // rsi
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r14
  __int64 v12; // r9
  unsigned __int64 v13; // r12
  __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned __int64 v16; // rdx
  __int64 v17; // r9
  unsigned __int64 v18; // rsi
  char *p_src; // rdi
  __int64 v20; // r15
  __int64 v21; // r13
  __int64 *v22; // rsi
  unsigned __int64 v23; // rax
  __int64 *v24; // r12
  unsigned __int64 v25; // r13
  __int64 v26; // r11
  char *v27; // rsi
  char *v28; // rdx
  __int64 v29; // r11
  __int64 v30; // rdx
  __int64 v31; // rsi
  unsigned __int64 v32; // rdx
  __int64 v33; // r11
  __int64 v34; // rdi
  unsigned __int64 v35; // rsi
  unsigned __int64 v36; // r15
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // r15
  char *v39; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v40; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v41; // [rsp+18h] [rbp-98h]
  __int64 v42; // [rsp+20h] [rbp-90h]
  char *v43; // [rsp+28h] [rbp-88h]
  unsigned __int64 v44; // [rsp+30h] [rbp-80h]
  __int64 v45; // [rsp+38h] [rbp-78h]
  __int64 src; // [rsp+40h] [rbp-70h] BYREF
  __int64 v47; // [rsp+48h] [rbp-68h] BYREF
  __int64 v48; // [rsp+50h] [rbp-60h]
  __int64 v49; // [rsp+58h] [rbp-58h]
  __int64 v50; // [rsp+60h] [rbp-50h]
  __int64 v51; // [rsp+68h] [rbp-48h]
  __int64 v52; // [rsp+70h] [rbp-40h]
  __int64 v53; // [rsp+78h] [rbp-38h]
  char v54[8]; // [rsp+80h] [rbp-30h] BYREF
  _BYTE v55[40]; // [rsp+88h] [rbp-28h] BYREF

  if ( a2 == a1 )
  {
    v7 = 0;
    return sub_AC25F0(&src, v7, (__int64)sub_C64CA0);
  }
  v3 = a1 + 4;
  v4 = &v47;
  v5 = v55;
  src = *a1;
  if ( a1 + 4 == a2 )
  {
LABEL_5:
    v7 = (char *)v4 - (char *)&src;
    return sub_AC25F0(&src, v7, (__int64)sub_C64CA0);
  }
  while ( ++v4 != (__int64 *)v55 )
  {
    v6 = *v3;
    v3 += 4;
    *(v4 - 1) = v6;
    if ( v3 == a2 )
      goto LABEL_5;
  }
  v9 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v10 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v44 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v10
           ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
           ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v10
          ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v47 + v9, 27));
  v11 = v51
      + v9
      - 0x4B6D499041670D8DLL * __ROL8__((char *)sub_C64CA0 + v52 + 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0, 22);
  v12 = src - 0x6D8ED9027DD26057LL * (_QWORD)sub_C64CA0;
  v13 = 0xB492B66FBE98F273LL
      * __ROL8__(
          v10
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
            ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
            ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
           ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
           ^ 0xB492B66FBE98F273LL))),
          31);
  v14 = v49 + v12 + v48 + v47;
  v15 = __ROR8__(v44 + v49 + v12 + v10, 21) + __ROL8__(v12 + v48 + v47, 20) + v12;
  v16 = v50
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (v10
          ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v10
         ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))))
      + v13;
  v45 = v15;
  v17 = 64;
  v18 = v16 + v51 + v52;
  p_src = (char *)&src;
  v20 = v53 + v18;
  v21 = __ROL8__(v18, 20) + __ROR8__(v16 + v11 + v48 + v53, 21);
  v22 = a2;
  v23 = v13;
  v24 = v22;
  v25 = v16 + v21;
  if ( v22 == v3 )
  {
    v27 = (char *)&src;
    goto LABEL_12;
  }
  while ( 1 )
  {
    v26 = *v3;
    v27 = (char *)&v47;
    v3 += 4;
    for ( src = v26; v24 != v3; *((_QWORD *)v28 - 1) = v29 )
    {
      v28 = v27 + 8;
      if ( v27 + 8 == v5 )
        break;
      v29 = *v3;
      v27 += 8;
      v3 += 4;
    }
    v17 += v27 - p_src;
LABEL_12:
    v39 = v5;
    v40 = v17;
    v41 = v23;
    v42 = v14;
    v43 = p_src;
    sub_F4F410(p_src, v27, v54);
    v30 = v11 + v42;
    v17 = v40;
    v5 = v39;
    v31 = 0xB492B66FBE98F273LL * v45 + src;
    v11 = v51 + v42 - 0x4B6D499041670D8DLL * __ROL8__(v52 + v45 + v11, 22);
    v23 = 0xB492B66FBE98F273LL * __ROL8__(v20 + v44, 31);
    v32 = v25 ^ (0xB492B66FBE98F273LL * __ROL8__(v41 + v47 + v30, 27));
    v33 = v31 + v48 + v47;
    v14 = v49 + v33;
    v34 = __ROR8__(v32 + v31 + v49 + v20, 21) + __ROL8__(v33, 20) + v31;
    v35 = v50 + v25 + v23;
    v45 = v34;
    p_src = v43;
    v20 = v53 + v35 + v51 + v52;
    v25 = __ROL8__(v35 + v51 + v52, 20) + v35 + __ROR8__(v11 + v35 + v48 + v53, 21);
    if ( v24 == v3 )
      break;
    v44 = v32;
  }
  v36 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v20 ^ v14)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v14)) ^ v20);
  v37 = 0xB492B66FBE98F273LL * ((v40 >> 47) ^ v40)
      + v23
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v25 ^ v45)) >> 47) ^ v25 ^ (0x9DDFEA08EB382D69LL * (v25 ^ v45)))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v25 ^ v45)) >> 47) ^ v25 ^ (0x9DDFEA08EB382D69LL * (v25 ^ v45)))));
  v38 = 0x9DDFEA08EB382D69LL
      * (v37 ^ (0xB492B66FBE98F273LL * ((v11 >> 47) ^ v11) + v32 - 0x622015F714C7D297LL * ((v36 >> 47) ^ v36)));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v38 >> 47) ^ v38 ^ v37)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v38 >> 47) ^ v38 ^ v37)));
}
