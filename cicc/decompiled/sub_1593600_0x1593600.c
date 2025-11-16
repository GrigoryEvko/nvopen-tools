// Function: sub_1593600
// Address: 0x1593600
//
unsigned __int64 __fastcall sub_1593600(_QWORD *a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  unsigned __int64 result; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rdx
  unsigned __int64 v12; // rdx
  __int64 v13; // r11
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // r9
  __int64 v17; // r13
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r11
  __int64 v21; // r9
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // r10
  __int64 v27; // rsi
  __int64 v28; // r9
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdx

  if ( a2 - 4 > 4 )
  {
    if ( a2 - 9 > 7 )
    {
      if ( a2 - 17 <= 0xF )
      {
        v25 = a1[1];
        v26 = 0x9AE16A3B2F90404FLL * *(_QWORD *)((char *)a1 + a2 - 8);
        v27 = __ROL8__(0xB492B66FBE98F273LL * *a1 - v25, 21) - 0x3C5A37A36834CED9LL * *(_QWORD *)((char *)a1 + a2 - 16);
        v28 = a3 - 0x4B6D499041670D8DLL * *a1 + a2 + __ROR8__(v25 ^ 0xC949D7C7509E6557LL, 20);
        v29 = __ROR8__(v26 ^ a3, 30);
        v30 = 0x9DDFEA08EB382D69LL
            * ((v28 - v26)
             ^ (0x9DDFEA08EB382D69LL * ((v28 - v26) ^ (v27 + v29)))
             ^ ((0x9DDFEA08EB382D69LL * ((v28 - v26) ^ (v27 + v29))) >> 47));
        return 0x9DDFEA08EB382D69LL * ((v30 >> 47) ^ v30);
      }
      else if ( a2 <= 0x20 )
      {
        result = a3 ^ 0x9AE16A3B2F90404FLL;
        if ( a2 )
        {
          v31 = a3
              ^ (0xC949D7C7509E6557LL * ((unsigned int)a2 + 4 * *((unsigned __int8 *)a1 + a2 - 1)))
              ^ (0x9AE16A3B2F90404FLL * (*(unsigned __int8 *)a1 + (*((unsigned __int8 *)a1 + (a2 >> 1)) << 8)));
          return 0x9AE16A3B2F90404FLL * (v31 ^ (v31 >> 47));
        }
      }
      else
      {
        v13 = *(_QWORD *)((char *)a1 + a2 - 16);
        v14 = a1[3];
        v15 = a1[2];
        v16 = *a1 - 0x3C5A37A36834CED9LL * (a2 + v13);
        v17 = v16 + a1[1];
        v18 = v17 + v15;
        v19 = *(_QWORD *)((char *)a1 + a2 - 32) + v15;
        v20 = v19 + *(_QWORD *)((char *)a1 + a2 - 24) + v13;
        v21 = __ROR8__(v18, 31) + __ROR8__(v17, 7) + __ROL8__(v14 + v16, 12) + __ROL8__(v16, 27);
        v22 = *(_QWORD *)((char *)a1 + a2 - 8);
        v23 = 0x9AE16A3B2F90404FLL
            * (v18
             + __ROR8__(v20, 31)
             + v14
             + __ROL8__(v19, 27)
             + __ROL8__(v19 + v22, 12)
             + __ROR8__(v19 + *(_QWORD *)((char *)a1 + a2 - 24), 7));
        v24 = v21
            + (a3
             ^ (0xC3A5C85C97CB3127LL
              * ((v23 - 0x3C5A37A36834CED9LL * (v21 + v22 + v20))
               ^ ((v23 - 0x3C5A37A36834CED9LL * (v21 + v22 + v20)) >> 47))));
        return 0x9AE16A3B2F90404FLL * (v24 ^ (v24 >> 47));
      }
    }
    else
    {
      v9 = *(_QWORD *)((char *)a1 + a2 - 8);
      v10 = *a1 ^ a3;
      v11 = __ROR8__(a2 + v9, a2);
      v12 = 0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v11 ^ v10)) ^ v11 ^ ((0x9DDFEA08EB382D69LL * (v11 ^ v10)) >> 47));
      return v9 ^ (0x9DDFEA08EB382D69LL * (v12 ^ (v12 >> 47)));
    }
  }
  else
  {
    v6 = a3 ^ *(unsigned int *)((char *)a1 + a2 - 4);
    v7 = 0x9DDFEA08EB382D69LL * (v6 ^ (a2 + 8LL * *(unsigned int *)a1));
    return 0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v7 ^ v6 ^ (v7 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v7 ^ v6 ^ (v7 >> 47))));
  }
  return result;
}
