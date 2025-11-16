// Function: sub_AC28A0
// Address: 0xac28a0
//
unsigned __int64 *__fastcall sub_AC28A0(unsigned __int64 *a1, __int64 *a2, unsigned __int64 a3)
{
  __int64 v6; // r11
  unsigned __int64 v7; // rcx
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // r9
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v16; // r9
  __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rcx
  unsigned __int64 v25; // rsi
  __int64 v26; // rcx
  unsigned __int64 v27; // r11

  v6 = a2[6];
  v7 = a3 ^ (a3 >> 47);
  v8 = a2[5];
  v9 = __ROL8__(a3 ^ 0xB492B66FBE98F273LL, 15);
  v10 = __ROL8__(0xB492B66FBE98F273LL * a3 + a3 + v6, 22);
  v11 = ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (a3 ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (a3 ^ 0xB492B66FBE98F273LL))
         ^ 0xB492B66FBE98F273LL)) >> 47)
      ^ (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (a3 ^ 0xB492B66FBE98F273LL)) >> 47)
        ^ (0x9DDFEA08EB382D69LL * (a3 ^ 0xB492B66FBE98F273LL))
        ^ 0xB492B66FBE98F273LL));
  v12 = 0x9DDFEA08EB382D69LL
      * (v7
       ^ (0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * a3)))
       ^ ((0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * a3))) >> 47));
  v13 = 0x9DDFEA08EB382D69LL * ((v12 >> 47) ^ v12);
  v14 = a2[1];
  v15 = __ROL8__(v9 + a3 + v14, 27);
  v16 = v8 + v9 - 0x4B6D499041670D8DLL * v10;
  v17 = a2[2];
  a1[1] = v16;
  v18 = v13 ^ (0xB492B66FBE98F273LL * v15);
  v19 = a2[4] + v13;
  a1[2] = v18;
  v20 = 0xB492B66FBE98F273LL * __ROL8__(v7 - 0x622015F714C7D297LL * v11, 31);
  v21 = *a2;
  *a1 = v20;
  v22 = 0x927126FD822D9FA9LL * a3 + v21;
  v23 = v22 + v17 + v14;
  v24 = __ROR8__(a2[3] + v22 + v7 + v18, 21);
  a1[3] = a2[3] + v23;
  v25 = v24 + v22 + __ROL8__(v23, 20);
  v26 = a2[7];
  v27 = v19 + v20 + v8 + v6;
  a1[4] = v25;
  a1[5] = v26 + v27;
  a1[6] = v19 + v20 + __ROL8__(v27, 20) + __ROR8__(v19 + v20 + v16 + v26 + v17, 21);
  return a1;
}
