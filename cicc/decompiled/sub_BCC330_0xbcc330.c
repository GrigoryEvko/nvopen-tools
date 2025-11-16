// Function: sub_BCC330
// Address: 0xbcc330
//
unsigned __int64 __fastcall sub_BCC330(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v5; // r13
  _QWORD *v6; // rbx
  __int64 v7; // r15
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // r13
  __int64 v15; // r9
  unsigned __int64 v16; // r8
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // rax
  __int64 v21; // rcx
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // r8
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // rax
  __int64 v32; // rcx
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // r14
  unsigned __int64 v35; // r9
  unsigned __int64 v36; // rcx
  __int64 v37; // r10
  unsigned __int64 v38; // rdi
  __int64 v39; // r13
  __int64 v40; // rdx
  __int64 v41; // rbx
  unsigned __int64 v42; // rdi
  __int64 v43; // r8
  __int64 v44; // rax
  __int64 v45; // r8
  unsigned __int64 v46; // rbx
  __int64 v47; // rax
  unsigned __int64 v48; // r10
  __int64 v49; // rcx

  v2 = a2 - (_QWORD)a1;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) <= 0x40 )
    return sub_AC25F0(a1, a2 - (_QWORD)a1, (__int64)sub_C64CA0);
  v5 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v6 = (_QWORD *)((char *)a1 + (v2 & 0xFFFFFFFFFFFFFFC0LL));
  v7 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v8 = 0x9DDFEA08EB382D69LL
     * (((0x9DDFEA08EB382D69LL
        * (v5
         ^ (0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ ((0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
      ^ (0x9DDFEA08EB382D69LL
       * (v5
        ^ (0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
        ^ ((0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))));
  v9 = a1[5] + v7 - 0x4B6D499041670D8DLL * __ROL8__(a1[6] - 0x4B6D499041670D8CLL * (_QWORD)sub_C64CA0, 22);
  v10 = v8 ^ (0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v7 + a1[1], 27));
  v11 = 0x927126FD822D9FA9LL * (_QWORD)sub_C64CA0 + *a1;
  v12 = 0xB492B66FBE98F273LL
      * __ROL8__(
          v5
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
  v13 = v5 + a1[3];
  v14 = v11 + a1[2] + a1[1];
  v15 = a1[3] + v14;
  v16 = v10 + v11 + v13;
  v17 = __ROL8__(v14, 20) + v11;
  v18 = a1[4] + v8 + v12;
  v19 = v17 + __ROR8__(v16, 21);
  v20 = v18 + a1[5] + a1[6];
  v21 = __ROL8__(v20, 20) + __ROR8__(v18 + v9 + a1[2] + a1[7], 21);
  while ( 1 )
  {
    v31 = a1[7] + v20;
    a1 += 8;
    v32 = v18 + v21;
    if ( v6 == a1 )
      break;
    v22 = v10;
    v23 = v9 + v15;
    v24 = __ROL8__(a1[6] + v19 + v9, 22);
    v25 = *a1 - 0x4B6D499041670D8DLL * v19;
    v26 = __ROL8__(v12 + a1[1] + v23, 27);
    v9 = a1[5] + v15 - 0x4B6D499041670D8DLL * v24;
    v12 = 0xB492B66FBE98F273LL * __ROL8__(v22 + v31, 31);
    v10 = v32 ^ (0xB492B66FBE98F273LL * v26);
    v27 = v25 + a1[2] + a1[1];
    v15 = a1[3] + v27;
    v28 = __ROR8__(v10 + v25 + a1[3] + v31, 21);
    v29 = __ROL8__(v27, 20) + v25;
    v30 = a1[4] + v32 + v12;
    v19 = v28 + v29;
    v20 = v30 + a1[5] + a1[6];
    v21 = __ROR8__(v9 + v30 + a1[2] + a1[7], 21);
    v18 = __ROL8__(v20, 20) + v30;
  }
  if ( (v2 & 0x3F) != 0 )
  {
    v37 = *(_QWORD *)(a2 - 16);
    v38 = v12 + v9;
    v39 = *(_QWORD *)(a2 - 24);
    v12 = 0xB492B66FBE98F273LL * __ROL8__(v31 + v10, 31);
    v40 = *(_QWORD *)(a2 - 48);
    v41 = *(_QWORD *)(a2 - 64) - 0x4B6D499041670D8DLL * v19;
    v9 = v39 + v15 - 0x4B6D499041670D8DLL * __ROL8__(v37 + v19 + v9, 22);
    v42 = v32 ^ (0xB492B66FBE98F273LL * __ROL8__(v15 + *(_QWORD *)(a2 - 56) + v38, 27));
    v43 = v41 + v40 + *(_QWORD *)(a2 - 56);
    v44 = __ROR8__(v41 + *(_QWORD *)(a2 - 40) + v31 + v42, 21);
    v15 = *(_QWORD *)(a2 - 40) + v43;
    v45 = v41 + __ROL8__(v43, 20);
    v46 = *(_QWORD *)(a2 - 32) + v32 + v12;
    v19 = v44 + v45;
    v47 = *(_QWORD *)(a2 - 8);
    v48 = v46 + v39 + v37;
    v49 = __ROR8__(v9 + v46 + v47 + v40, 21);
    v31 = v48 + v47;
    v10 = v42;
    v32 = __ROL8__(v48, 20) + v46 + v49;
  }
  v33 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v32 ^ v19)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v32 ^ v19)) ^ v32);
  v34 = 0x9DDFEA08EB382D69LL * ((v33 >> 47) ^ v33) + 0xB492B66FBE98F273LL * (v2 ^ (v2 >> 47)) + v12;
  v35 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v31 ^ v15)) >> 47) ^ v31 ^ (0x9DDFEA08EB382D69LL * (v31 ^ v15)));
  v36 = 0x9DDFEA08EB382D69LL
      * (v34 ^ (0xB492B66FBE98F273LL * ((v9 >> 47) ^ v9) + v10 - 0x622015F714C7D297LL * ((v35 >> 47) ^ v35)));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v36 ^ v34 ^ (v36 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v36 ^ v34 ^ (v36 >> 47))));
}
