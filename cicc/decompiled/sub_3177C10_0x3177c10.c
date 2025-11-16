// Function: sub_3177C10
// Address: 0x3177c10
//
unsigned __int64 __fastcall sub_3177C10(_QWORD *a1, _QWORD *a2)
{
  __int64 *v2; // r13
  _QWORD *v3; // r15
  __int64 v4; // r9
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // r9
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  _QWORD *v10; // r14
  unsigned __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // r10
  __int64 v16; // rdi
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  __int64 v19; // r8
  __int64 v20; // rdx
  unsigned __int64 v21; // r11
  unsigned __int64 v22; // rdx
  __int64 v23; // r9
  __int64 v24; // rsi
  unsigned __int64 v25; // r8
  _QWORD *v26; // rax
  char *v27; // r14
  _QWORD *v28; // r15
  __int64 v29; // rdi
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rdi
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rax
  char *v35; // rsi
  unsigned __int64 v37; // r8
  unsigned __int64 v38; // rax
  char *v39; // [rsp+0h] [rbp-F0h]
  unsigned __int64 v40; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v41; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v42; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v43; // [rsp+20h] [rbp-D0h]
  __int64 v44; // [rsp+28h] [rbp-C8h]
  __int64 v45; // [rsp+30h] [rbp-C0h]
  __int64 v46; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v47; // [rsp+40h] [rbp-B0h]
  unsigned __int64 v49; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int64 v50; // [rsp+78h] [rbp-78h] BYREF
  __int64 v51; // [rsp+80h] [rbp-70h] BYREF
  __int64 v52; // [rsp+88h] [rbp-68h]
  __int64 v53; // [rsp+90h] [rbp-60h]
  __int64 v54; // [rsp+98h] [rbp-58h]
  __int64 v55; // [rsp+A0h] [rbp-50h]
  __int64 v56; // [rsp+A8h] [rbp-48h]
  __int64 v57; // [rsp+B0h] [rbp-40h]
  __int64 v58; // [rsp+B8h] [rbp-38h]
  char v59[8]; // [rsp+C0h] [rbp-30h] BYREF
  _BYTE v60[40]; // [rsp+C8h] [rbp-28h] BYREF

  if ( a1 == a2 )
    return sub_AC25F0(&v51, 0, (__int64)sub_C64CA0);
  v2 = &v51;
  v3 = a1;
  while ( 1 )
  {
    ++v2;
    v4 = HIDWORD(v3[1]);
    v5 = 0x9DDFEA08EB382D69LL * (v4 ^ ((unsigned __int64)sub_C64CA0 + ((8LL * v3[1]) & 0x7FFFFFFF8LL)));
    v6 = (v5 >> 47) ^ v5 ^ v4;
    v7 = HIDWORD(*v3);
    v8 = 0x9DDFEA08EB382D69LL * (v7 ^ ((unsigned __int64)sub_C64CA0 + ((8LL * *v3) & 0x7FFFFFFF8LL)));
    v49 = 0x9DDFEA08EB382D69LL * (((0x9DDFEA08EB382D69LL * v6) >> 47) ^ (0x9DDFEA08EB382D69LL * v6));
    v50 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * ((v8 >> 47) ^ v8 ^ v7)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v8 >> 47) ^ v8 ^ v7)));
    v9 = sub_C41E80((__int64 *)&v50, (__int64 *)&v49);
    if ( v2 == (__int64 *)v60 )
      break;
    *(v2 - 1) = v9;
    v3 += 2;
    if ( a2 == v3 )
      return sub_AC25F0(&v51, (char *)v2 - (char *)&v51, (__int64)sub_C64CA0);
  }
  v10 = v3;
  v11 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v12 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v40 = 64;
  v41 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v11
           ^ (0x9DDFEA08EB382D69LL * (v11 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
           ^ ((0x9DDFEA08EB382D69LL * (v11 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v11
          ^ (0x9DDFEA08EB382D69LL * (v11 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v11 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v52 + v12, 27));
  v13 = v51 - 0x6D8ED9027DD26057LL * (_QWORD)sub_C64CA0;
  v14 = 0xB492B66FBE98F273LL
      * __ROL8__(
          v11
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
  v15 = 0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v57 + 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0, 22)
      + v56
      + v12;
  v16 = v13 + v53 + v52;
  v47 = v15;
  v46 = v54 + v16;
  v17 = v14
      + v55
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (v11
          ^ (0x9DDFEA08EB382D69LL * (v11 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v11 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v11
         ^ (0x9DDFEA08EB382D69LL * (v11 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ ((0x9DDFEA08EB382D69LL * (v11 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))));
  v43 = v14;
  v44 = __ROR8__(v41 + v54 + v13 + v11, 21) + __ROL8__(v16, 20) + v13;
  v18 = v17 + v56 + v57;
  v45 = v58 + v18;
  v42 = __ROL8__(v18, 20) + __ROR8__(v17 + v15 + v53 + v58, 21) + v17;
  while ( 1 )
  {
    v26 = v10;
    v27 = (char *)&v51;
    v28 = v26;
    do
    {
      v29 = HIDWORD(v28[1]);
      v30 = 0x9DDFEA08EB382D69LL * (v29 ^ ((unsigned __int64)sub_C64CA0 + ((8LL * v28[1]) & 0x7FFFFFFF8LL)));
      v31 = (v30 >> 47) ^ v30 ^ v29;
      v32 = HIDWORD(*v28);
      v33 = 0x9DDFEA08EB382D69LL * (v32 ^ ((unsigned __int64)sub_C64CA0 + ((8LL * *v28) & 0x7FFFFFFF8LL)));
      v49 = 0x9DDFEA08EB382D69LL * (((0x9DDFEA08EB382D69LL * v31) >> 47) ^ (0x9DDFEA08EB382D69LL * v31));
      v50 = 0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * ((v33 >> 47) ^ v32 ^ v33)) >> 47)
           ^ (0x9DDFEA08EB382D69LL * ((v33 >> 47) ^ v32 ^ v33)));
      v34 = sub_C41E80((__int64 *)&v50, (__int64 *)&v49);
      v35 = v27;
      v27 += 8;
      if ( v27 == v60 )
      {
        v10 = v28;
        goto LABEL_6;
      }
      *((_QWORD *)v27 - 1) = v34;
      v28 += 2;
    }
    while ( a2 != v28 );
    v35 = v27;
    v10 = v28;
LABEL_6:
    v39 = v35;
    sub_3174C40((char *)&v51, v35, v59);
    v19 = v51 - 0x4B6D499041670D8DLL * v44;
    v20 = __ROL8__(v43 + v52 + v46 + v47, 27);
    v21 = 0xB492B66FBE98F273LL * __ROL8__(v57 + v44 + v47, 22) + v56 + v46;
    v47 = v21;
    v22 = v42 ^ (0xB492B66FBE98F273LL * v20);
    v43 = 0xB492B66FBE98F273LL * __ROL8__(v45 + v41, 31);
    v23 = v19 + v53 + v52;
    v46 = v54 + v23;
    v24 = __ROR8__(v22 + v19 + v54 + v45, 21) + __ROL8__(v23, 20) + v19;
    v25 = v43 + v55 + v42;
    v44 = v24;
    v40 += v39 - (char *)&v51;
    v42 = __ROL8__(v25 + v56 + v57, 20) + v25 + __ROR8__(v21 + v25 + v53 + v58, 21);
    v45 = v25 + v56 + v57 + v58;
    if ( a2 == v10 )
      break;
    v41 = v22;
  }
  v37 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v42 ^ v24)) >> 47) ^ v42 ^ (0x9DDFEA08EB382D69LL * (v42 ^ v24)))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v42 ^ v24)) >> 47) ^ v42 ^ (0x9DDFEA08EB382D69LL * (v42 ^ v24)))))
      + v43
      - 0x4B6D499041670D8DLL * (v40 ^ (v40 >> 47));
  v38 = 0x9DDFEA08EB382D69LL
      * (v37
       ^ (0xB492B66FBE98F273LL * (v21 ^ (v21 >> 47))
        + v22
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v45 ^ v46)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v45 ^ v46)) ^ v45)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v45 ^ v46)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v45 ^ v46)) ^ v45)))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v38 ^ v37 ^ (v38 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v38 ^ v37 ^ (v38 >> 47))));
}
