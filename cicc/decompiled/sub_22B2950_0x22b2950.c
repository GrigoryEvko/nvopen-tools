// Function: sub_22B2950
// Address: 0x22b2950
//
unsigned __int64 __fastcall sub_22B2950(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // r8
  _QWORD *v5; // rax
  __int64 v6; // r8
  _QWORD *v7; // rax
  __int64 v8; // r8
  char *v9; // rax
  __int64 v10; // r14
  signed __int64 v12; // rbx
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-B8h] BYREF
  __int64 v16; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v17; // [rsp+18h] [rbp-A8h] BYREF
  _OWORD dest[4]; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v19; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 v20; // [rsp+68h] [rbp-58h]
  __int64 v21; // [rsp+70h] [rbp-50h]
  __int64 v22; // [rsp+78h] [rbp-48h]
  __int64 v23; // [rsp+80h] [rbp-40h]
  __int64 v24; // [rsp+88h] [rbp-38h]
  __int64 v25; // [rsp+90h] [rbp-30h]
  void (__fastcall *v26)(__int64, __int64); // [rsp+98h] [rbp-28h]

  v4 = *a1;
  memset(dest, 0, sizeof(dest));
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = sub_C64CA0;
  v15 = 0;
  v5 = sub_CA5190((unsigned __int64 *)dest, &v15, dest, (unsigned __int64)&v19, v4);
  v6 = *a2;
  v16 = v15;
  v7 = sub_CA5190((unsigned __int64 *)dest, &v16, v5, (unsigned __int64)&v19, v6);
  v8 = *a3;
  v17 = v16;
  v9 = (char *)sub_CA5190((unsigned __int64 *)dest, &v17, v7, (unsigned __int64)&v19, v8);
  v10 = v17;
  if ( !v17 )
    return sub_AC25F0(dest, v9 - (char *)dest, (__int64)v26);
  v12 = v9 - (char *)dest;
  sub_22AD600((char *)dest, v9, (char *)&v19);
  sub_AC2A10(&v19, dest);
  v13 = v19
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v25 ^ v23)) ^ v25 ^ ((0x9DDFEA08EB382D69LL * (v25 ^ v23)) >> 47)))
       ^ ((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v25 ^ v23)) ^ v25 ^ ((0x9DDFEA08EB382D69LL * (v25 ^ v23)) >> 47))) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v10 + v12) >> 47) ^ (v10 + v12));
  v14 = 0x9DDFEA08EB382D69LL
      * (v13
       ^ (v21
        - 0x4B6D499041670D8DLL * (v20 ^ (v20 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v24 ^ v22)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v24 ^ v22)) ^ v24)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v24 ^ v22)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v24 ^ v22)) ^ v24)))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v14 ^ v13 ^ (v14 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v14 ^ v13 ^ (v14 >> 47))));
}
