// Function: sub_2B479F0
// Address: 0x2b479f0
//
unsigned __int64 __fastcall sub_2B479F0(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v5; // r13
  _QWORD *v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  _QWORD *v14; // rdi
  __int64 v15; // rax
  unsigned __int64 v16; // r14
  unsigned __int64 v17; // r8
  __int64 v18; // r13
  __int64 v19; // r9
  unsigned __int64 v20; // r8
  __int64 v21; // rax
  unsigned __int64 v22; // r13
  __int64 v23; // rcx
  __int64 v24; // r8
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // r15
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // r13
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // r8
  unsigned __int64 v37; // r13
  __int64 v38; // rcx
  __int64 v39; // r15
  __int64 v40; // r12
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // [rsp+0h] [rbp-70h] BYREF
  unsigned __int64 v43; // [rsp+8h] [rbp-68h]
  unsigned __int64 v44; // [rsp+10h] [rbp-60h]
  __int64 v45; // [rsp+18h] [rbp-58h]
  __int64 v46; // [rsp+20h] [rbp-50h]
  unsigned __int64 v47; // [rsp+28h] [rbp-48h]
  unsigned __int64 v48; // [rsp+30h] [rbp-40h]

  v2 = a2 - (_QWORD)a1;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) <= 0x40 )
    return sub_AC25F0(a1, a2 - (_QWORD)a1, (__int64)sub_C64CA0);
  v5 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v6 = (__int64 *)((char *)a1 + (v2 & 0xFFFFFFFFFFFFFFC0LL));
  v7 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v8 = __ROL8__((char *)sub_C64CA0 + v7 + a1[1], 27);
  v9 = a1[5] + v7 - 0x4B6D499041670D8DLL * __ROL8__(a1[6] - 0x4B6D499041670D8CLL * (_QWORD)sub_C64CA0, 22);
  v43 = v9;
  v10 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v5
           ^ (0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
           ^ ((0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v5
          ^ (0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * v8);
  v11 = a1[2] + a1[1];
  v12 = *a1;
  v13 = a1[4]
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (v5
          ^ (0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v5
         ^ (0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ ((0x9DDFEA08EB382D69LL * (v5 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))));
  v14 = a1 + 8;
  v44 = v10;
  v15 = 0x927126FD822D9FA9LL * (_QWORD)sub_C64CA0 + v12;
  v16 = 0xB492B66FBE98F273LL
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
  v17 = v5 + *(v14 - 5);
  v18 = v15 + v11;
  v19 = *(v14 - 5) + v15 + v11;
  v45 = v19;
  v20 = v10 + v15 + v17;
  v21 = __ROL8__(v18, 20) + v15;
  v22 = v13 + v16;
  v23 = *(v14 - 6) + *(v14 - 1);
  v42 = v16;
  v24 = v21 + __ROR8__(v20, 21);
  v25 = v22 + *(v14 - 3) + *(v14 - 2);
  v46 = v24;
  v26 = v25;
  v27 = *(v14 - 1) + v25;
  v47 = v27;
  v28 = v22 + __ROL8__(v26, 20) + __ROR8__(v22 + v9 + v23, 21);
  v48 = v28;
  if ( v6 != v14 )
  {
    do
    {
      v29 = v10;
      v30 = v9 + v19;
      v31 = __ROL8__(v14[6] + v24 + v9, 22);
      v32 = *v14 - 0x4B6D499041670D8DLL * v24;
      v33 = __ROL8__(v16 + v14[1] + v30, 27);
      v9 = v14[5] + v19 - 0x4B6D499041670D8DLL * v31;
      v16 = 0xB492B66FBE98F273LL * __ROL8__(v29 + v27, 31);
      v10 = v28 ^ (0xB492B66FBE98F273LL * v33);
      v34 = v32 + v14[2] + v14[1];
      v19 = v14[3] + v34;
      v35 = __ROR8__(v10 + v32 + v14[3] + v27, 21);
      v36 = __ROL8__(v34, 20) + v32;
      v37 = v14[4] + v28 + v16;
      v24 = v35 + v36;
      v38 = __ROR8__(v37 + v9 + v14[2] + v14[7], 21);
      v39 = __ROL8__(v37 + v14[5] + v14[6], 20);
      v27 = v14[7] + v37 + v14[5] + v14[6];
      v14 += 8;
      v28 = v39 + v37 + v38;
    }
    while ( v6 != v14 );
    v42 = v16;
    v43 = v9;
    v45 = v19;
    v46 = v24;
    v48 = v28;
    v44 = v10;
    v47 = v27;
  }
  if ( (v2 & 0x3F) != 0 )
    sub_AC2A10(&v42, (_QWORD *)(a2 - 64));
  v40 = v42
      + 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v48 ^ v46)) ^ v48 ^ ((0x9DDFEA08EB382D69LL * (v48 ^ v46)) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v48 ^ v46)) ^ v48 ^ ((0x9DDFEA08EB382D69LL * (v48 ^ v46)) >> 47))))
      - 0x4B6D499041670D8DLL * ((v2 >> 47) ^ v2);
  v41 = 0x9DDFEA08EB382D69LL
      * (v40
       ^ (v44
        - 0x4B6D499041670D8DLL * (v43 ^ (v43 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v47 ^ v45)) ^ v47 ^ ((0x9DDFEA08EB382D69LL * (v47 ^ v45)) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v47 ^ v45)) ^ v47 ^ ((0x9DDFEA08EB382D69LL * (v47 ^ v45)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v41 ^ v40 ^ (v41 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v41 ^ v40 ^ (v41 >> 47))));
}
