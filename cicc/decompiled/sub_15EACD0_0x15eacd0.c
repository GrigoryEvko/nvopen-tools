// Function: sub_15EACD0
// Address: 0x15eacd0
//
unsigned __int64 __fastcall sub_15EACD0(char *a1, char *a2)
{
  char *v2; // r10
  char *v3; // rcx
  unsigned __int64 v4; // r12
  char v5; // di
  char *v6; // rsi
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rdx
  unsigned __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // r12
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // r14
  char *v22; // rsi
  unsigned __int64 v23; // r10
  char *v24; // rbx
  __int64 v25; // r14
  __int64 v26; // rcx
  __int64 v27; // r12
  __int64 v28; // r15
  char *v29; // rdx
  __int64 v30; // rax
  char *v31; // r14
  __int64 v32; // r10
  __int64 v33; // rdi
  unsigned __int64 v34; // r9
  __int64 v35; // r15
  __int64 v36; // rcx
  __int64 v37; // rcx
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int64 v41; // rdi
  char *i; // rsi
  char *v43; // r13
  __int64 v44; // r14
  __int64 v45; // rax
  unsigned __int64 v46; // r15
  unsigned __int64 v47; // r12
  unsigned __int64 v48; // r8
  unsigned __int64 v49; // r11
  unsigned __int64 v50; // r8
  unsigned __int64 v51; // rsi
  int v53; // eax
  unsigned __int64 v54; // rax
  __int64 v55; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v56; // [rsp+10h] [rbp-B0h]
  __int64 v57; // [rsp+18h] [rbp-A8h]
  __int64 v58; // [rsp+20h] [rbp-A0h]
  char *v59; // [rsp+28h] [rbp-98h]
  unsigned __int64 v60; // [rsp+38h] [rbp-88h]
  unsigned __int64 v61; // [rsp+40h] [rbp-80h]
  __int64 v62; // [rsp+50h] [rbp-70h] BYREF
  __int64 v63; // [rsp+58h] [rbp-68h]
  __int64 v64; // [rsp+60h] [rbp-60h]
  __int64 v65; // [rsp+68h] [rbp-58h]
  __int64 v66; // [rsp+70h] [rbp-50h]
  __int64 v67; // [rsp+78h] [rbp-48h]
  __int64 v68; // [rsp+80h] [rbp-40h]
  __int64 v69; // [rsp+88h] [rbp-38h]
  char v70; // [rsp+90h] [rbp-30h] BYREF
  char v71; // [rsp+91h] [rbp-2Fh] BYREF

  v2 = a2;
  v3 = a1;
  if ( !byte_4F99930[0] )
  {
    v53 = sub_2207590(byte_4F99930);
    v3 = a1;
    v2 = a2;
    if ( v53 )
    {
      v54 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v54 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v54;
      sub_2207640(byte_4F99930);
      v2 = a2;
      v3 = a1;
    }
  }
  v4 = qword_4F99938;
  if ( v2 == a1 )
    return sub_1593600(&v62, 0, qword_4F99938);
  v5 = *a1;
  v6 = (char *)&v62 + 1;
  do
  {
    ++v3;
    *(v6 - 1) = v5;
    if ( v3 == v2 )
      return sub_1593600(&v62, v6 - (char *)&v62, v4);
    ++v6;
    v5 = *v3;
  }
  while ( v6 != &v71 );
  v60 = 64;
  v7 = __ROL8__(v4 ^ 0xB492B66FBE98F273LL, 15);
  v8 = v4 ^ (v4 >> 47);
  v61 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v8
           ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4)))
           ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v8
          ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4)))
          ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * __ROL8__(v7 + v4 + v63, 27));
  v9 = v67 + v7 - 0x4B6D499041670D8DLL * __ROL8__(v4 + v68 - 0x4B6D499041670D8DLL * v4, 22);
  v10 = v66
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (v8
          ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4)))
          ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v8
         ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4)))
         ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4))) >> 47))));
  v11 = v62 - 0x6D8ED9027DD26057LL * v4;
  v12 = 0xB492B66FBE98F273LL
      * __ROL8__(
          v8
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL)) >> 47)
            ^ (0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL))
            ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL)) >> 47)
           ^ (0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL))
           ^ 0xB492B66FBE98F273LL))),
          31);
  v13 = v11 + v64 + v63;
  v14 = v65 + v13;
  v15 = __ROR8__(v65 + v11 + v8 + v61, 21);
  v16 = v11 + __ROL8__(v13, 20);
  v17 = v10 + v12;
  v18 = v15 + v16;
  v19 = v10 + v12 + v67 + v68;
  v20 = v69 + v19;
  v21 = __ROL8__(v19, 20) + __ROR8__(v10 + v12 + v9 + v64 + v69, 21);
  v22 = v2;
  v23 = v12;
  v24 = v3;
  v25 = v17 + v21;
  v26 = v18;
  v27 = v9;
  v28 = v20;
  v29 = &v70;
  v30 = v25;
  v31 = v22;
  while ( 1 )
  {
    for ( i = (char *)&v62 + 1; ; ++i )
    {
      v43 = v24 + 1;
      *(i - 1) = v5;
      if ( v31 == v24 + 1 || v29 == i )
        break;
      v5 = *v43;
      ++v24;
    }
    v55 = v30;
    v56 = v23;
    v57 = v26;
    v58 = v14;
    v59 = v29;
    sub_15EA400((char *)&v62, i, v29);
    v32 = __ROL8__(v63 + v27 + v58 + v56, 27);
    v33 = v62 - 0x4B6D499041670D8DLL * v57;
    v27 = v67 + v58 - 0x4B6D499041670D8DLL * __ROL8__(v68 + v57 + v27, 22);
    v34 = v55 ^ (0xB492B66FBE98F273LL * v32);
    v23 = 0xB492B66FBE98F273LL * __ROL8__(v28 + v61, 31);
    v35 = __ROR8__(v34 + v33 + v28 + v65, 21);
    v36 = v33 + v64 + v63;
    v14 = v65 + v36;
    v37 = v33 + __ROL8__(v36, 20);
    v38 = v66 + v55 + v23;
    v26 = v35 + v37;
    v60 += i - (char *)&v62;
    v39 = v38 + v67 + v68;
    v40 = __ROR8__(v27 + v38 + v64 + v69, 21);
    v28 = v69 + v39;
    v41 = __ROL8__(v39, 20) + v38;
    v29 = v59;
    v30 = v41 + v40;
    if ( v31 == v43 )
      break;
    v61 = v34;
    v5 = *++v24;
  }
  v44 = v30;
  v45 = v28;
  v46 = v27;
  v47 = ((0x9DDFEA08EB382D69LL * (v44 ^ v26)) >> 47) ^ v44 ^ (0x9DDFEA08EB382D69LL * (v44 ^ v26));
  v48 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v45 ^ v14)) >> 47) ^ v45 ^ (0x9DDFEA08EB382D69LL * (v45 ^ v14)));
  v49 = 0xB492B66FBE98F273LL * (v46 ^ (v46 >> 47));
  v50 = 0x9DDFEA08EB382D69LL * ((v48 >> 47) ^ v48);
  v51 = 0x9DDFEA08EB382D69LL * (((0x9DDFEA08EB382D69LL * v47) >> 47) ^ (0x9DDFEA08EB382D69LL * v47))
      + v23
      - 0x4B6D499041670D8DLL * (v60 ^ (v60 >> 47));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v51 ^ (v49 + v34 + v50))) >> 47)
           ^ v51
           ^ (0x9DDFEA08EB382D69LL * (v51 ^ (v49 + v34 + v50))))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v51 ^ (v49 + v34 + v50))) >> 47)
          ^ v51
          ^ (0x9DDFEA08EB382D69LL * (v51 ^ (v49 + v34 + v50))))));
}
