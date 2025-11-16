// Function: sub_AF6940
// Address: 0xaf6940
//
unsigned __int64 __fastcall sub_AF6940(__int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // r13
  __int64 *v6; // r12
  __int64 *v7; // rdi
  unsigned __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // r10
  __int64 v11; // r8
  __int64 v12; // r15
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r11
  unsigned __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // r15
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // r13
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v27; // [rsp+18h] [rbp-68h]
  unsigned __int64 v28; // [rsp+20h] [rbp-60h]
  __int64 v29; // [rsp+28h] [rbp-58h]
  __int64 v30; // [rsp+30h] [rbp-50h]
  unsigned __int64 v31; // [rsp+38h] [rbp-48h]
  __int64 v32; // [rsp+40h] [rbp-40h]

  v3 = a2 - (_QWORD)a1;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) <= 0x40 )
    return sub_AC25F0(a1, a2 - (_QWORD)a1, (__int64)sub_C64CA0);
  v6 = (__int64 *)((char *)a1 + (v3 & 0xFFFFFFFFFFFFFFC0LL));
  sub_AC28A0(&v26, a1, (unsigned __int64)sub_C64CA0);
  v7 = a1 + 8;
  if ( v6 != a1 + 8 )
  {
    v8 = v26;
    v9 = v27;
    v10 = v29;
    v11 = v30;
    v12 = v32;
    v13 = v28;
    v14 = v31;
    do
    {
      v15 = v13;
      v16 = v10 + v9;
      v17 = __ROL8__(v7[6] + v11 + v9, 22);
      v18 = *v7 - 0x4B6D499041670D8DLL * v11;
      v19 = __ROL8__(v8 + v7[1] + v16, 27);
      v9 = v7[5] + v10 - 0x4B6D499041670D8DLL * v17;
      v8 = 0xB492B66FBE98F273LL * __ROL8__(v15 + v14, 31);
      v20 = v18 + v7[2] + v7[1];
      v13 = v12 ^ (0xB492B66FBE98F273LL * v19);
      v10 = v7[3] + v20;
      v11 = __ROR8__(v13 + v18 + v7[3] + v14, 21) + __ROL8__(v20, 20) + v18;
      v21 = v12 + v7[4] + v8;
      v22 = __ROR8__(v21 + v9 + v7[2] + v7[7], 21);
      v23 = __ROL8__(v21 + v7[5] + v7[6], 20);
      v14 = v7[7] + v21 + v7[5] + v7[6];
      v7 += 8;
      v12 = v22 + v23 + v21;
    }
    while ( v6 != v7 );
    v26 = v8;
    v27 = v9;
    v29 = v10;
    v30 = v11;
    v32 = v12;
    v28 = v13;
    v31 = v14;
  }
  if ( (v3 & 0x3F) != 0 )
    sub_AC2A10(&v26, (_QWORD *)(a2 - 64));
  v24 = v26
      + 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v32 ^ v30)) ^ v32 ^ ((0x9DDFEA08EB382D69LL * (v32 ^ v30)) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v32 ^ v30)) ^ v32 ^ ((0x9DDFEA08EB382D69LL * (v32 ^ v30)) >> 47))))
      - 0x4B6D499041670D8DLL * ((v3 >> 47) ^ v3);
  v25 = 0x9DDFEA08EB382D69LL
      * (v24
       ^ (v28
        - 0x4B6D499041670D8DLL * (v27 ^ (v27 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v31 ^ v29)) ^ v31 ^ ((0x9DDFEA08EB382D69LL * (v31 ^ v29)) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v31 ^ v29)) ^ v31 ^ ((0x9DDFEA08EB382D69LL * (v31 ^ v29)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v25 ^ v24 ^ (v25 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v25 ^ v24 ^ (v25 >> 47))));
}
