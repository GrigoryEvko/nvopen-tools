// Function: sub_27B2C30
// Address: 0x27b2c30
//
unsigned __int64 __fastcall sub_27B2C30(int *a1, __int64 *a2, _DWORD *a3, _BYTE *a4, __int64 a5)
{
  int v5; // eax
  __int64 *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  char *v10; // rax
  __int64 v11; // r14
  signed __int64 v13; // rbx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-A8h] BYREF
  int dest; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v18; // [rsp+14h] [rbp-9Ch]
  int v19; // [rsp+1Ch] [rbp-94h]
  _OWORD v20[3]; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v21; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v22; // [rsp+58h] [rbp-58h]
  __int64 v23; // [rsp+60h] [rbp-50h]
  __int64 v24; // [rsp+68h] [rbp-48h]
  __int64 v25; // [rsp+70h] [rbp-40h]
  __int64 v26; // [rsp+78h] [rbp-38h]
  __int64 v27; // [rsp+80h] [rbp-30h]
  void (__fastcall *v28)(__int64, __int64); // [rsp+88h] [rbp-28h]

  memset(v20, 0, sizeof(v20));
  v28 = sub_C64CA0;
  v5 = *a1;
  v6 = *(__int64 **)a5;
  dest = v5;
  v7 = *a2;
  v21 = 0;
  v18 = v7;
  LODWORD(v7) = *a3;
  v22 = 0;
  v19 = v7;
  LOBYTE(v7) = *a4;
  v23 = 0;
  LOBYTE(v20[0]) = v7;
  v8 = *(_QWORD *)(a5 + 8);
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v16 = 0;
  v9 = sub_AC61D0(v6, (__int64)v6 + 4 * v8);
  v10 = (char *)sub_CA5190((unsigned __int64 *)&dest, &v16, (_OWORD *)((char *)v20 + 1), (unsigned __int64)&v21, v9);
  v11 = v16;
  if ( !v16 )
    return sub_AC25F0(&dest, v10 - (char *)&dest, (__int64)v28);
  v13 = v10 - (char *)&dest;
  sub_27AC2B0((char *)&dest, v10, (char *)&v21);
  sub_AC2A10(&v21, &dest);
  v14 = v21
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v27 ^ v25)) ^ v27 ^ ((0x9DDFEA08EB382D69LL * (v27 ^ v25)) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v27 ^ v25)) ^ v27 ^ ((0x9DDFEA08EB382D69LL * (v27 ^ v25)) >> 47))))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v11 + v13) >> 47) ^ (v11 + v13));
  v15 = 0x9DDFEA08EB382D69LL
      * (v14
       ^ (0xB492B66FBE98F273LL * (v22 ^ (v22 >> 47))
        + v23
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v26 ^ v24)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v26 ^ v24)) ^ v26)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v26 ^ v24)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v26 ^ v24)) ^ v26)))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v15 ^ v14 ^ (v15 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v15 ^ v14 ^ (v15 >> 47))));
}
