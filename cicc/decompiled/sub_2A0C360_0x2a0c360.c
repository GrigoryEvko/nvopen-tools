// Function: sub_2A0C360
// Address: 0x2a0c360
//
unsigned __int64 __fastcall sub_2A0C360(__int64 *a1, _QWORD *a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rax
  char *v5; // rsi
  _QWORD *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r13
  __int64 v11; // r10
  unsigned __int64 v12; // rbx
  __int64 v13; // r8
  __int64 v14; // rax
  unsigned __int64 v15; // r10
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // r12
  _QWORD *v20; // rdx
  unsigned __int64 v21; // r12
  __int64 v22; // rax
  _QWORD *v23; // rsi
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r15
  unsigned __int64 v26; // r10
  _QWORD *v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rsi
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  char *v34; // r11
  char *v35; // rsi
  _QWORD *v36; // rdx
  _QWORD *v37; // rdx
  _QWORD *v38; // rdx
  unsigned __int64 v40; // r10
  unsigned __int64 v41; // rdx
  __int64 v42; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v43; // [rsp+8h] [rbp-98h]
  unsigned __int64 v44; // [rsp+10h] [rbp-90h]
  unsigned __int64 v45; // [rsp+20h] [rbp-80h]
  __int64 v46; // [rsp+28h] [rbp-78h]
  __int64 v47; // [rsp+30h] [rbp-70h] BYREF
  __int64 v48; // [rsp+38h] [rbp-68h]
  __int64 v49; // [rsp+40h] [rbp-60h]
  __int64 v50; // [rsp+48h] [rbp-58h]
  __int64 v51; // [rsp+50h] [rbp-50h]
  __int64 v52; // [rsp+58h] [rbp-48h]
  __int64 v53; // [rsp+60h] [rbp-40h]
  __int64 v54; // [rsp+68h] [rbp-38h]
  char v55[48]; // [rsp+70h] [rbp-30h] BYREF

  v3 = 0;
  v4 = *a1;
  if ( *a1 == *a2 )
    return sub_AC25F0(&v47, v3, (__int64)sub_C64CA0);
  v5 = (char *)&v47;
  while ( 1 )
  {
    v5 += 8;
    v6 = (_QWORD *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v4 & 4) != 0 )
      break;
    v7 = v6[17];
    if ( v5 > v55 )
      goto LABEL_5;
    v38 = v6 + 18;
    *((_QWORD *)v5 - 1) = v7;
    *a1 = (__int64)v38;
    v4 = (__int64)v38;
LABEL_20:
    if ( v38 == (_QWORD *)*a2 )
    {
      v3 = v5 - (char *)&v47;
      return sub_AC25F0(&v47, v3, (__int64)sub_C64CA0);
    }
  }
  if ( v5 <= v55 )
  {
    *((_QWORD *)v5 - 1) = *(_QWORD *)(*v6 + 136LL);
    v4 = (unsigned __int64)(v6 + 1) | 4;
    *a1 = v4;
    v38 = (_QWORD *)v4;
    goto LABEL_20;
  }
LABEL_5:
  v8 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v9 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v45 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v9
           ^ (0x9DDFEA08EB382D69LL * (v9 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
           ^ ((0x9DDFEA08EB382D69LL * (v9 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v9
          ^ (0x9DDFEA08EB382D69LL * (v9 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v9 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v48 + v8, 27));
  v10 = v52
      + v8
      - 0x4B6D499041670D8DLL * __ROL8__((char *)sub_C64CA0 + v53 + 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0, 22);
  v11 = v47 - 0x6D8ED9027DD26057LL * (_QWORD)sub_C64CA0;
  v12 = 0xB492B66FBE98F273LL
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
        + v9,
          31);
  v13 = v50 + v11 + v49 + v48;
  v14 = __ROR8__(v45 + v50 + v11 + v9, 21) + __ROL8__(v11 + v49 + v48, 20) + v11;
  v15 = 0x24AD9BEFA63C9CC0LL;
  v46 = v14;
  v16 = v51
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (v9
          ^ (0x9DDFEA08EB382D69LL * (v9 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v9 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v9
         ^ (0x9DDFEA08EB382D69LL * (v9 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ ((0x9DDFEA08EB382D69LL * (v9 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))))
      + v12;
  v17 = v16 + v52 + v53;
  v18 = v54 + v17;
  v19 = __ROL8__(v17, 20) + __ROR8__(v16 + v10 + v49 + v54, 21);
  v20 = (_QWORD *)*a2;
  v21 = v16 + v19;
  v22 = *a1;
  if ( *a2 != *a1 )
  {
    v23 = a2;
    v24 = v21;
    v25 = 64;
    v26 = v12;
    v27 = v23;
    while ( 1 )
    {
      v34 = (char *)&v47;
      v35 = (char *)&v47;
      if ( v20 != (_QWORD *)v22 )
      {
        do
        {
          v35 = v34;
          v34 += 8;
          v36 = (_QWORD *)(v22 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v22 & 4) != 0 )
          {
            if ( v34 > v55 )
            {
LABEL_8:
              v25 += v35 - (char *)&v47;
              goto LABEL_9;
            }
            *((_QWORD *)v34 - 1) = *(_QWORD *)(*v36 + 136LL);
            v22 = (unsigned __int64)(v36 + 1) | 4;
            *a1 = v22;
            v37 = (_QWORD *)v22;
          }
          else
          {
            v28 = v36[17];
            if ( v34 > v55 )
              goto LABEL_8;
            v37 = v36 + 18;
            *((_QWORD *)v34 - 1) = v28;
            *a1 = (__int64)v37;
            v22 = (__int64)v37;
          }
        }
        while ( v37 != (_QWORD *)*v27 );
        v35 = v34;
        v25 += v34 - (char *)&v47;
      }
LABEL_9:
      v42 = v13;
      v43 = v24;
      v44 = v26;
      sub_2A0B400((char *)&v47, v35, v55);
      v29 = v10 + v44;
      v30 = v47 - 0x4B6D499041670D8DLL * v46;
      v10 = v52 + v42 - 0x4B6D499041670D8DLL * __ROL8__(v53 + v46 + v10, 22);
      v26 = 0xB492B66FBE98F273LL * __ROL8__(v18 + v45, 31);
      v31 = v43 ^ (0xB492B66FBE98F273LL * __ROL8__(v42 + v48 + v29, 27));
      v32 = v30 + v49 + v48;
      v13 = v50 + v32;
      v46 = __ROR8__(v31 + v30 + v50 + v18, 21) + __ROL8__(v32, 20) + v30;
      v33 = v51 + v43 + v26;
      v18 = v54 + v33 + v52 + v53;
      v20 = (_QWORD *)*v27;
      v24 = __ROL8__(v33 + v52 + v53, 20) + v33 + __ROR8__(v10 + v33 + v49 + v54, 21);
      v22 = *a1;
      if ( *a1 == *v27 )
        break;
      v45 = v31;
    }
    v12 = v26;
    v45 = v31;
    v21 = v24;
    v15 = 0xB492B66FBE98F273LL * ((v25 >> 47) ^ v25);
  }
  v40 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v21 ^ v46)) >> 47) ^ v21 ^ (0x9DDFEA08EB382D69LL * (v21 ^ v46)))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v21 ^ v46)) >> 47) ^ v21 ^ (0x9DDFEA08EB382D69LL * (v21 ^ v46)))))
      + v12
      + v15;
  v41 = 0x9DDFEA08EB382D69LL
      * (v40
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v18 ^ v13)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v18 ^ v13)) ^ v18)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v18 ^ v13)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v18 ^ v13)) ^ v18)))
        + 0xB492B66FBE98F273LL * ((v10 >> 47) ^ v10)
        + v45));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v41 ^ v40 ^ (v41 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v41 ^ v40 ^ (v41 >> 47))));
}
