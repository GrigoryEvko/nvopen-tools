// Function: sub_AF1490
// Address: 0xaf1490
//
unsigned __int64 __fastcall sub_AF1490(_QWORD *a1, unsigned __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rcx
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rdx

  v2 = a1[6];
  v3 = 0x9DDFEA08EB382D69LL * (v2 ^ a1[4]);
  v4 = *a1
     - 0x4B6D499041670D8DLL * ((a2 >> 47) ^ a2)
     - 0x622015F714C7D297LL
     * (((0x9DDFEA08EB382D69LL * (v3 ^ v2 ^ (v3 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v3 ^ v2 ^ (v3 >> 47))));
  v5 = 0x9DDFEA08EB382D69LL
     * (v4
      ^ (a1[2]
       - 0x4B6D499041670D8DLL * (a1[1] ^ (a1[1] >> 47))
       - 0x622015F714C7D297LL
       * (((0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (a1[5] ^ a1[3])) ^ a1[5] ^ ((0x9DDFEA08EB382D69LL * (a1[5] ^ a1[3])) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (a1[5] ^ a1[3])) ^ a1[5] ^ ((0x9DDFEA08EB382D69LL * (a1[5] ^ a1[3])) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v5 >> 47) ^ v4 ^ v5)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v5 >> 47) ^ v4 ^ v5)));
}
