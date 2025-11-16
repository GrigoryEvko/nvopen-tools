// Function: sub_22B0AE0
// Address: 0x22b0ae0
//
unsigned __int64 __fastcall sub_22B0AE0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  _QWORD *v3; // r12
  _QWORD *v6; // rsi
  __int64 v7; // rax
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // [rsp+0h] [rbp-70h] BYREF
  void (__fastcall *v10)(__int64, __int64); // [rsp+8h] [rbp-68h]
  unsigned __int64 v11; // [rsp+10h] [rbp-60h]
  __int64 v12; // [rsp+18h] [rbp-58h]
  unsigned __int64 v13; // [rsp+20h] [rbp-50h]
  unsigned __int64 v14; // [rsp+28h] [rbp-48h]
  unsigned __int64 v15; // [rsp+30h] [rbp-40h]

  v2 = a2 - (_QWORD)a1;
  v3 = a1;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) <= 0x40 )
    return sub_AC25F0(a1, a2 - (_QWORD)a1, (__int64)sub_C64CA0);
  v10 = sub_C64CA0;
  v9 = 0;
  v12 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v13 = 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0;
  v11 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
          ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
          ^ 0xB492B66FBE98F273LL)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
         ^ 0xB492B66FBE98F273LL)));
  v14 = ((unsigned __int64)sub_C64CA0 >> 47) ^ (unsigned __int64)sub_C64CA0;
  v15 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v14 ^ v13)) >> 47) ^ v14 ^ (0x9DDFEA08EB382D69LL * (v14 ^ v13)))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v14 ^ v13)) >> 47) ^ v14 ^ (0x9DDFEA08EB382D69LL * (v14 ^ v13)))));
  do
  {
    v6 = v3;
    v3 += 8;
    sub_AC2A10(&v9, v6);
  }
  while ( (_QWORD *)((char *)a1 + (v2 & 0xFFFFFFFFFFFFFFC0LL)) != v3 );
  if ( (v2 & 0x3F) != 0 )
    sub_AC2A10(&v9, (_QWORD *)(a2 - 64));
  v7 = v9
     + 0xB492B66FBE98F273LL * ((v2 >> 47) ^ v2)
     - 0x622015F714C7D297LL
     * (((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v15 ^ v13)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v15 ^ v13)) ^ v15)) >> 47)
      ^ (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v15 ^ v13)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v15 ^ v13)) ^ v15)));
  v8 = 0x9DDFEA08EB382D69LL
     * (v7
      ^ (v11
       - 0x4B6D499041670D8DLL * ((unsigned __int64)v10 ^ ((unsigned __int64)v10 >> 47))
       - 0x622015F714C7D297LL
       * (((0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v14 ^ v12)) ^ v14 ^ ((0x9DDFEA08EB382D69LL * (v14 ^ v12)) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v14 ^ v12)) ^ v14 ^ ((0x9DDFEA08EB382D69LL * (v14 ^ v12)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v8 >> 47) ^ v8 ^ v7)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v8 >> 47) ^ v8 ^ v7)));
}
