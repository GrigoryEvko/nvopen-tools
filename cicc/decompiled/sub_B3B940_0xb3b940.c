// Function: sub_B3B940
// Address: 0xb3b940
//
unsigned __int64 __fastcall sub_B3B940(char *a1, char *a2)
{
  char *v3; // rbx
  char *v4; // rsi
  char *v5; // r9
  char v6; // al
  unsigned __int64 v7; // rsi
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rcx
  __int64 v13; // r10
  unsigned __int64 v14; // r12
  __int64 v15; // rsi
  __int64 v16; // r14
  char *v17; // rdi
  __int64 v18; // r10
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // r14
  signed __int64 v22; // r11
  unsigned __int64 v23; // r10
  unsigned __int64 v24; // r12
  __int64 v25; // rcx
  __int64 v26; // rdi
  __int64 v27; // rsi
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rcx
  char v30; // dl
  char *v31; // rax
  char *v32; // rsi
  char v33; // dl
  unsigned __int64 v34; // rcx
  unsigned __int64 v35; // r14
  unsigned __int64 v36; // rax
  signed __int64 v37; // [rsp+10h] [rbp-B0h]
  char *v38; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v39; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v40; // [rsp+28h] [rbp-98h]
  char *src; // [rsp+30h] [rbp-90h]
  __int64 v42; // [rsp+38h] [rbp-88h]
  __int64 v43; // [rsp+40h] [rbp-80h]
  unsigned __int64 v44; // [rsp+48h] [rbp-78h]
  __int64 v45; // [rsp+50h] [rbp-70h] BYREF
  __int64 v46; // [rsp+58h] [rbp-68h]
  __int64 v47; // [rsp+60h] [rbp-60h]
  __int64 v48; // [rsp+68h] [rbp-58h]
  __int64 v49; // [rsp+70h] [rbp-50h]
  __int64 v50; // [rsp+78h] [rbp-48h]
  __int64 v51; // [rsp+80h] [rbp-40h]
  __int64 v52; // [rsp+88h] [rbp-38h]
  char v53[48]; // [rsp+90h] [rbp-30h] BYREF

  if ( a2 == a1 )
  {
    v7 = 0;
    return sub_AC25F0(&v45, v7, (__int64)sub_C64CA0);
  }
  v3 = a1 + 1;
  v4 = (char *)&v45 + 1;
  v5 = v53;
  LOBYTE(v45) = *a1;
  if ( a1 + 1 == a2 )
  {
LABEL_5:
    v7 = v4 - (char *)&v45;
    return sub_AC25F0(&v45, v7, (__int64)sub_C64CA0);
  }
  while ( v4 != v53 )
  {
    v6 = *v3++;
    *v4++ = v6;
    if ( v3 == a2 )
      goto LABEL_5;
  }
  v9 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v10 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v11 = v50
      + v9
      - 0x4B6D499041670D8DLL * __ROL8__((char *)sub_C64CA0 + v51 + 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0, 22);
  v12 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v10
           ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
           ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v10
          ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v46 + v9, 27));
  v13 = v45 - 0x6D8ED9027DD26057LL * (_QWORD)sub_C64CA0;
  v14 = 0xB492B66FBE98F273LL
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
  v15 = v13 + v47 + v46;
  v16 = __ROR8__(v12 + v48 + v10 + v13, 21);
  v42 = v48 + v15;
  v17 = (char *)&v45;
  v18 = __ROL8__(v15, 20) + v13;
  v19 = v49
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (v10
          ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v10
         ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))))
      + v14;
  v43 = v16 + v18;
  v20 = v19 + v50 + v51;
  v21 = v52 + v20;
  v22 = v53 - (char *)&v45;
  v44 = __ROL8__(v20, 20) + __ROR8__(v19 + v11 + v47 + v52, 21) + v19;
  v23 = v14;
  v24 = 64;
  while ( 1 )
  {
    if ( a2 == v3 )
    {
      v32 = v17;
    }
    else
    {
      v30 = *v3;
      v31 = (char *)&v45 + 1;
      ++v3;
      LOBYTE(v45) = v30;
      v32 = (char *)&v45 + 1;
      if ( a2 == v3 )
      {
LABEL_15:
        v24 += v31 - v17;
      }
      else
      {
        while ( v32 != v5 )
        {
          v33 = *v3;
          ++v31;
          ++v3;
          v32 = v31;
          *(v31 - 1) = v33;
          if ( a2 == v3 )
            goto LABEL_15;
        }
        v24 += v22;
      }
    }
    v37 = v22;
    v38 = v5;
    v39 = v23;
    v40 = v12;
    src = v17;
    sub_B3AFC0(v17, v32, v53);
    v25 = v45 - 0x4B6D499041670D8DLL * v43;
    v26 = v50 + v42;
    v27 = v25 + v47 + v46;
    v5 = v38;
    v28 = v44 ^ (0xB492B66FBE98F273LL * __ROL8__(v39 + v46 + v11 + v42, 27));
    v23 = 0xB492B66FBE98F273LL * __ROL8__(v40 + v21, 31);
    v42 = v48 + v27;
    v11 = v26 - 0x4B6D499041670D8DLL * __ROL8__(v51 + v43 + v11, 22);
    v43 = __ROR8__(v28 + v25 + v48 + v21, 21) + __ROL8__(v27, 20) + v25;
    v22 = v37;
    v29 = v23 + v44 + v49;
    v21 = v52 + v29 + v50 + v51;
    v44 = __ROR8__(v11 + v29 + v47 + v52, 21) + __ROL8__(v29 + v50 + v51, 20) + v29;
    v17 = src;
    if ( a2 == v3 )
      break;
    v12 = v28;
  }
  v34 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v44 ^ v43)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v44 ^ v43)) ^ v44)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v44 ^ v43)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v44 ^ v43)) ^ v44)))
      + 0xB492B66FBE98F273LL * ((v24 >> 47) ^ v24)
      + v23;
  v35 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v21 ^ v42)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v21 ^ v42)) ^ v21);
  v36 = 0x9DDFEA08EB382D69LL
      * (v34
       ^ (0x9DDFEA08EB382D69LL
        * (v34 ^ (0x9DDFEA08EB382D69LL * ((v35 >> 47) ^ v35) + 0xB492B66FBE98F273LL * (v11 ^ (v11 >> 47)) + v28)))
       ^ ((0x9DDFEA08EB382D69LL
         * (v34 ^ (0x9DDFEA08EB382D69LL * ((v35 >> 47) ^ v35) + 0xB492B66FBE98F273LL * (v11 ^ (v11 >> 47)) + v28))) >> 47));
  return 0x9DDFEA08EB382D69LL * ((v36 >> 47) ^ v36);
}
