// Function: sub_939680
// Address: 0x939680
//
unsigned __int64 __fastcall sub_939680(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r8
  unsigned __int64 v3; // rcx
  __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // r8
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r13
  __int64 v17; // r15
  _QWORD *v18; // r12
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // r9
  unsigned __int64 v25; // r13
  __int64 v26; // r10
  unsigned __int64 v27; // r9
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // r13
  unsigned __int64 v30; // r9
  unsigned __int64 v31; // rax
  __int64 v32; // rsi
  unsigned __int64 v33; // r13
  unsigned __int64 v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // r9
  unsigned __int64 v41; // r13
  unsigned __int64 v42; // rax
  __int64 v43; // rsi
  unsigned __int64 v44; // rsi
  unsigned __int64 v45; // r10
  unsigned __int64 v46; // r14
  unsigned __int64 v47; // rcx
  __int64 v48; // rsi
  __int64 v49; // rdx
  unsigned __int64 v50; // rdx
  __int64 v51; // r11
  unsigned __int64 v52; // r8
  __int64 v53; // r13
  __int64 v54; // rdx
  __int64 v55; // r12
  unsigned __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rax
  __int64 v59; // r9
  unsigned __int64 v60; // r12
  __int64 v61; // rax
  unsigned __int64 v62; // r11
  __int64 v63; // rsi
  unsigned __int64 v64; // r9
  __int64 v65; // r11
  __int64 v66; // r12
  __int64 v67; // r10
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rax
  __int64 v71; // rsi
  __int64 v72; // r8
  __int64 v73; // rcx
  __int64 v74; // rsi
  unsigned __int64 v75; // rdx
  unsigned __int64 v76; // rdx

  v2 = a1;
  v3 = a2 - (_QWORD)a1;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) > 0x40 )
  {
    v16 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
    v17 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
    v18 = (_QWORD *)((char *)a1 + (v3 & 0xFFFFFFFFFFFFFFC0LL));
    v19 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL
           * (v16
            ^ (0x9DDFEA08EB382D69LL * (v16 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
            ^ ((0x9DDFEA08EB382D69LL * (v16 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (v16
           ^ (0x9DDFEA08EB382D69LL * (v16 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
           ^ ((0x9DDFEA08EB382D69LL * (v16 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))));
    v20 = v19 ^ (0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v17 + a1[1], 27));
    v21 = a1[5] + v17 - 0x4B6D499041670D8DLL * __ROL8__(a1[6] - 0x4B6D499041670D8CLL * (_QWORD)sub_C64CA0, 22);
    v22 = 0x927126FD822D9FA9LL * (_QWORD)sub_C64CA0 + *v2;
    v23 = 0xB492B66FBE98F273LL
        * __ROL8__(
            v16
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
    v24 = v16 + v2[3];
    v25 = v22 + v2[2] + v2[1];
    v26 = v2[3] + v25;
    v27 = v20 + v22 + v24;
    v28 = __ROL8__(v25, 20) + v22;
    v29 = v2[4] + v19 + v23;
    v30 = v28 + __ROR8__(v27, 21);
    v31 = v29 + v2[5] + v2[6];
    v32 = __ROL8__(v31, 20) + __ROR8__(v29 + v21 + v2[2] + v2[7], 21);
    while ( 1 )
    {
      v42 = v2[7] + v31;
      v2 += 8;
      v43 = v29 + v32;
      if ( v18 == v2 )
        break;
      v33 = v20;
      v34 = v21 + v26;
      v35 = __ROL8__(v2[6] + v30 + v21, 22);
      v36 = *v2 - 0x4B6D499041670D8DLL * v30;
      v37 = __ROL8__(v23 + v2[1] + v34, 27);
      v21 = v2[5] + v26 - 0x4B6D499041670D8DLL * v35;
      v23 = 0xB492B66FBE98F273LL * __ROL8__(v33 + v42, 31);
      v20 = v43 ^ (0xB492B66FBE98F273LL * v37);
      v38 = v36 + v2[2] + v2[1];
      v26 = v2[3] + v38;
      v39 = __ROR8__(v20 + v36 + v2[3] + v42, 21);
      v40 = __ROL8__(v38, 20) + v36;
      v41 = v2[4] + v43 + v23;
      v30 = v39 + v40;
      v31 = v41 + v2[5] + v2[6];
      v32 = __ROR8__(v21 + v41 + v2[2] + v2[7], 21);
      v29 = __ROL8__(v31, 20) + v41;
    }
    if ( (v3 & 0x3F) != 0 )
    {
      v51 = *(_QWORD *)(a2 - 16);
      v52 = v23 + v21;
      v53 = *(_QWORD *)(a2 - 24);
      v23 = 0xB492B66FBE98F273LL * __ROL8__(v42 + v20, 31);
      v54 = *(_QWORD *)(a2 - 48);
      v55 = *(_QWORD *)(a2 - 64) - 0x4B6D499041670D8DLL * v30;
      v21 = v53 + v26 - 0x4B6D499041670D8DLL * __ROL8__(v51 + v30 + v21, 22);
      v56 = v43 ^ (0xB492B66FBE98F273LL * __ROL8__(v26 + *(_QWORD *)(a2 - 56) + v52, 27));
      v57 = v55 + v54 + *(_QWORD *)(a2 - 56);
      v58 = __ROR8__(v55 + *(_QWORD *)(a2 - 40) + v42 + v56, 21);
      v26 = *(_QWORD *)(a2 - 40) + v57;
      v59 = v55 + __ROL8__(v57, 20);
      v60 = *(_QWORD *)(a2 - 32) + v43 + v23;
      v30 = v58 + v59;
      v61 = *(_QWORD *)(a2 - 8);
      v62 = v60 + v53 + v51;
      v63 = __ROR8__(v21 + v60 + v61 + v54, 21);
      v42 = v62 + v61;
      v20 = v56;
      v43 = __ROL8__(v62, 20) + v60 + v63;
    }
    v44 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v43 ^ v30)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v43 ^ v30)) ^ v43);
    v45 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v42 ^ v26)) >> 47) ^ v42 ^ (0x9DDFEA08EB382D69LL * (v42 ^ v26)));
    v46 = 0x9DDFEA08EB382D69LL * ((v44 >> 47) ^ v44) + 0xB492B66FBE98F273LL * (v3 ^ (v3 >> 47)) + v23;
    v47 = 0x9DDFEA08EB382D69LL
        * (v46 ^ (0xB492B66FBE98F273LL * ((v21 >> 47) ^ v21) + v20 - 0x622015F714C7D297LL * ((v45 >> 47) ^ v45)));
    return 0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v47 ^ v46 ^ (v47 >> 47))) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v47 ^ v46 ^ (v47 >> 47))));
  }
  else if ( v3 - 4 > 4 )
  {
    if ( v3 - 9 <= 7 )
    {
      v48 = *(_QWORD *)(a2 - 8);
      v49 = __ROR8__(v3 + v48, v3);
      v50 = 0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v49 ^ *a1 ^ (unsigned __int64)sub_C64CA0))
           ^ v49
           ^ ((0x9DDFEA08EB382D69LL * (v49 ^ *a1 ^ (unsigned __int64)sub_C64CA0)) >> 47));
      return v48 ^ (0x9DDFEA08EB382D69LL * (v50 ^ (v50 >> 47)));
    }
    else if ( v3 - 17 > 0xF )
    {
      if ( v3 > 0x20 )
      {
        v65 = *(_QWORD *)(a2 - 16);
        v66 = a1[3];
        v67 = *(_QWORD *)(a2 - 8);
        v68 = v3 + v65;
        v69 = a1[2];
        v70 = *a1 - 0x3C5A37A36834CED9LL * v68;
        v71 = v70 + a1[1];
        v72 = v71 + v69;
        v73 = *(_QWORD *)(a2 - 32) + v69;
        v74 = __ROL8__(v66 + v70, 12) + __ROR8__(v71, 7) + __ROL8__(v70, 27) + __ROR8__(v72, 31);
        v75 = 0x9AE16A3B2F90404FLL
            * (v72
             + __ROR8__(v73 + *(_QWORD *)(a2 - 24) + v65, 31)
             + v66
             + __ROL8__(v73, 27)
             + __ROL8__(v73 + v67, 12)
             + __ROR8__(v73 + *(_QWORD *)(a2 - 24), 7))
            - 0x3C5A37A36834CED9LL * (v74 + v67 + v73 + *(_QWORD *)(a2 - 24) + v65);
        v76 = ((0xC3A5C85C97CB3127LL * ((v75 >> 47) ^ v75)) ^ (unsigned __int64)sub_C64CA0) + v74;
        return 0x9AE16A3B2F90404FLL * (v76 ^ (v76 >> 47));
      }
      else
      {
        result = (unsigned __int64)sub_C64CA0 ^ 0x9AE16A3B2F90404FLL;
        if ( v3 )
        {
          v64 = (0xC949D7C7509E6557LL * ((unsigned int)v3 + 4 * *(unsigned __int8 *)(a2 - 1)))
              ^ (0x9AE16A3B2F90404FLL * (*(unsigned __int8 *)a1 + (*((unsigned __int8 *)a1 + (v3 >> 1)) << 8)))
              ^ (unsigned __int64)sub_C64CA0;
          return 0x9AE16A3B2F90404FLL * (v64 ^ (v64 >> 47));
        }
      }
    }
    else
    {
      v9 = 0xB492B66FBE98F273LL * *a1;
      v10 = a1[1];
      v11 = 0x9AE16A3B2F90404FLL * *(_QWORD *)(a2 - 8);
      v12 = (unsigned __int64)sub_C64CA0 + v9 + v3 + __ROR8__(v10 ^ 0xC949D7C7509E6557LL, 20) - v11;
      v13 = 0xC3A5C85C97CB3127LL * *(_QWORD *)(a2 - 16) + __ROL8__(v9 - v10, 21);
      v14 = __ROR8__(v11 ^ (unsigned __int64)sub_C64CA0, 30);
      v15 = 0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v12 ^ (v13 + v14))) ^ v12 ^ ((0x9DDFEA08EB382D69LL * (v12 ^ (v13 + v14))) >> 47));
      return 0x9DDFEA08EB382D69LL * (v15 ^ (v15 >> 47));
    }
  }
  else
  {
    v5 = *(unsigned int *)(a2 - 4);
    v6 = 0x9DDFEA08EB382D69LL * (v5 ^ (unsigned __int64)sub_C64CA0 ^ (v3 + 8LL * *(unsigned int *)a1));
    v7 = 0x9DDFEA08EB382D69LL * (v6 ^ v5 ^ (unsigned __int64)sub_C64CA0 ^ (v6 >> 47));
    return 0x9DDFEA08EB382D69LL * (v7 ^ (v7 >> 47));
  }
  return result;
}
