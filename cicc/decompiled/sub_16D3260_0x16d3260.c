// Function: sub_16D3260
// Address: 0x16d3260
//
unsigned __int64 __fastcall sub_16D3260(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // r8
  unsigned __int64 v6; // rcx
  unsigned __int64 result; // rax
  unsigned __int64 v8; // rsi
  __int64 v9; // r9
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // r9
  __int64 v16; // r15
  _QWORD *v17; // r13
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // r8
  unsigned __int64 v24; // r9
  __int64 v25; // r10
  unsigned __int64 v26; // r8
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // r9
  unsigned __int64 v29; // r8
  unsigned __int64 v30; // r15
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rsi
  _QWORD *i; // r9
  unsigned __int64 v34; // rbx
  unsigned __int64 v35; // rdx
  __int64 v36; // rdi
  __int64 v37; // r8
  __int64 v38; // rdx
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // r8
  unsigned __int64 v42; // rbx
  __int64 v43; // rsi
  __int64 v44; // r15
  unsigned __int64 v45; // rsi
  unsigned __int64 v46; // r10
  unsigned __int64 v47; // r14
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rdx
  unsigned __int64 v52; // rdx
  __int64 v53; // r11
  unsigned __int64 v54; // r9
  __int64 v55; // r13
  __int64 v56; // r14
  __int64 v57; // rdx
  __int64 v58; // rbx
  unsigned __int64 v59; // r9
  __int64 v60; // r8
  __int64 v61; // rax
  __int64 v62; // r8
  unsigned __int64 v63; // rbx
  __int64 v64; // rax
  unsigned __int64 v65; // r11
  __int64 v66; // rsi
  unsigned __int64 v67; // r8
  __int64 v68; // r11
  __int64 v69; // r13
  __int64 v70; // r10
  __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rax
  __int64 v74; // rsi
  __int64 v75; // r9
  __int64 v76; // rcx
  __int64 v77; // rsi
  unsigned __int64 v78; // rdx
  unsigned __int64 v79; // rdx

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v49 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v49 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v49;
    sub_2207640(byte_4F99930);
  }
  v4 = a2 - (_QWORD)a1;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) > 0x40 )
  {
    v15 = qword_4F99938 ^ ((unsigned __int64)qword_4F99938 >> 47);
    v16 = __ROL8__(qword_4F99938 ^ 0xB492B66FBE98F273LL, 15);
    v17 = (_QWORD *)((char *)a1 + (v4 & 0xFFFFFFFFFFFFFFC0LL));
    v18 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL
           * (v15
            ^ (0x9DDFEA08EB382D69LL * (v15 ^ (0xB492B66FBE98F273LL * qword_4F99938)))
            ^ ((0x9DDFEA08EB382D69LL * (v15 ^ (0xB492B66FBE98F273LL * qword_4F99938))) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (v15
           ^ (0x9DDFEA08EB382D69LL * (v15 ^ (0xB492B66FBE98F273LL * qword_4F99938)))
           ^ ((0x9DDFEA08EB382D69LL * (v15 ^ (0xB492B66FBE98F273LL * qword_4F99938))) >> 47))));
    v19 = a1[5] + v16 - 0x4B6D499041670D8DLL * __ROL8__(a1[6] - 0x4B6D499041670D8CLL * qword_4F99938, 22);
    v20 = v18 ^ (0xB492B66FBE98F273LL * __ROL8__(a1[1] + qword_4F99938 + v16, 27));
    v21 = 0x927126FD822D9FA9LL * qword_4F99938 + *a1;
    v22 = 0xB492B66FBE98F273LL
        * __ROL8__(
            v15
          - 0x622015F714C7D297LL
          * (((0x9DDFEA08EB382D69LL
             * (((0x9DDFEA08EB382D69LL * (qword_4F99938 ^ 0xB492B66FBE98F273LL)) >> 47)
              ^ (0x9DDFEA08EB382D69LL * (qword_4F99938 ^ 0xB492B66FBE98F273LL))
              ^ 0xB492B66FBE98F273LL)) >> 47)
           ^ (0x9DDFEA08EB382D69LL
            * (((0x9DDFEA08EB382D69LL * (qword_4F99938 ^ 0xB492B66FBE98F273LL)) >> 47)
             ^ (0x9DDFEA08EB382D69LL * (qword_4F99938 ^ 0xB492B66FBE98F273LL))
             ^ 0xB492B66FBE98F273LL))),
            31);
    v23 = v15 + a1[3];
    v24 = v21 + a1[2] + a1[1];
    v25 = a1[3] + v24;
    v26 = v20 + v21 + v23;
    v27 = __ROL8__(v24, 20) + v21;
    v28 = a1[4] + v18 + v22;
    v29 = v27 + __ROR8__(v26, 21);
    v30 = v28 + a1[5] + a1[6];
    v31 = a1[7] + v30;
    v32 = v28 + __ROL8__(v30, 20) + __ROR8__(v28 + v19 + a1[2] + a1[7], 21);
    for ( i = a1 + 8; v17 != i; v32 = v44 + v42 + v43 )
    {
      v34 = v20;
      v35 = v19 + v25;
      v36 = __ROL8__(i[6] + v29 + v19, 22);
      v37 = *i - 0x4B6D499041670D8DLL * v29;
      v38 = __ROL8__(v22 + i[1] + v35, 27);
      v19 = i[5] + v25 - 0x4B6D499041670D8DLL * v36;
      v22 = 0xB492B66FBE98F273LL * __ROL8__(v34 + v31, 31);
      v20 = v32 ^ (0xB492B66FBE98F273LL * v38);
      v39 = v37 + i[2] + i[1];
      v25 = i[3] + v39;
      v40 = __ROR8__(v20 + v37 + i[3] + v31, 21);
      v41 = __ROL8__(v39, 20) + v37;
      v42 = i[4] + v32 + v22;
      v29 = v40 + v41;
      v43 = __ROR8__(v19 + v42 + i[2] + i[7], 21);
      v44 = __ROL8__(v42 + i[5] + i[6], 20);
      v31 = i[7] + v42 + i[5] + i[6];
      i += 8;
    }
    if ( (v4 & 0x3F) != 0 )
    {
      v53 = *(_QWORD *)(a2 - 16);
      v54 = v22 + v19;
      v55 = *(_QWORD *)(a2 - 24);
      v56 = __ROL8__(v20 + v31, 31);
      v57 = *(_QWORD *)(a2 - 48);
      v22 = 0xB492B66FBE98F273LL * v56;
      v19 = v55 + v25 - 0x4B6D499041670D8DLL * __ROL8__(v53 + v29 + v19, 22);
      v58 = *(_QWORD *)(a2 - 64) - 0x4B6D499041670D8DLL * v29;
      v59 = v32 ^ (0xB492B66FBE98F273LL * __ROL8__(v25 + *(_QWORD *)(a2 - 56) + v54, 27));
      v60 = v58 + v57 + *(_QWORD *)(a2 - 56);
      v61 = __ROR8__(v58 + *(_QWORD *)(a2 - 40) + v31 + v59, 21);
      v25 = *(_QWORD *)(a2 - 40) + v60;
      v62 = v58 + __ROL8__(v60, 20);
      v63 = *(_QWORD *)(a2 - 32) + v32 + v22;
      v29 = v61 + v62;
      v64 = *(_QWORD *)(a2 - 8);
      v65 = v63 + v55 + v53;
      v66 = v64 + v57;
      v20 = v59;
      v32 = __ROL8__(v65, 20) + v63 + __ROR8__(v19 + v63 + v66, 21);
      v31 = v65 + v64;
    }
    v45 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v32 ^ v29)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v32 ^ v29)) ^ v32);
    v46 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v31 ^ v25)) >> 47) ^ v31 ^ (0x9DDFEA08EB382D69LL * (v31 ^ v25)));
    v47 = 0x9DDFEA08EB382D69LL * ((v45 >> 47) ^ v45) + 0xB492B66FBE98F273LL * (v4 ^ (v4 >> 47)) + v22;
    v48 = 0x9DDFEA08EB382D69LL
        * (v47 ^ (v20 - 0x4B6D499041670D8DLL * ((v19 >> 47) ^ v19) - 0x622015F714C7D297LL * ((v46 >> 47) ^ v46)));
    return 0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v48 ^ v47 ^ (v48 >> 47))) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v48 ^ v47 ^ (v48 >> 47))));
  }
  else if ( v4 - 4 > 4 )
  {
    if ( v4 - 9 <= 7 )
    {
      v50 = *(_QWORD *)(a2 - 8);
      v51 = __ROR8__(v4 + v50, v4);
      v52 = 0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v51 ^ *a1 ^ qword_4F99938))
           ^ v51
           ^ ((0x9DDFEA08EB382D69LL * (v51 ^ *a1 ^ qword_4F99938)) >> 47));
      return v50 ^ (0x9DDFEA08EB382D69LL * (v52 ^ (v52 >> 47)));
    }
    else if ( v4 - 17 > 0xF )
    {
      if ( v4 > 0x20 )
      {
        v68 = *(_QWORD *)(a2 - 16);
        v69 = a1[3];
        v70 = *(_QWORD *)(a2 - 8);
        v71 = v4 + v68;
        v72 = a1[2];
        v73 = *a1 - 0x3C5A37A36834CED9LL * v71;
        v74 = v73 + a1[1];
        v75 = v74 + v72;
        v76 = *(_QWORD *)(a2 - 32) + v72;
        v77 = __ROL8__(v69 + v73, 12) + __ROR8__(v74, 7) + __ROL8__(v73, 27) + __ROR8__(v75, 31);
        v78 = 0x9AE16A3B2F90404FLL
            * (v75
             + __ROR8__(v76 + *(_QWORD *)(a2 - 24) + v68, 31)
             + v69
             + __ROL8__(v76, 27)
             + __ROL8__(v76 + v70, 12)
             + __ROR8__(v76 + *(_QWORD *)(a2 - 24), 7))
            - 0x3C5A37A36834CED9LL * (v77 + v70 + v76 + *(_QWORD *)(a2 - 24) + v68);
        v79 = ((0xC3A5C85C97CB3127LL * ((v78 >> 47) ^ v78)) ^ qword_4F99938) + v77;
        return 0x9AE16A3B2F90404FLL * (v79 ^ (v79 >> 47));
      }
      else
      {
        result = qword_4F99938 ^ 0x9AE16A3B2F90404FLL;
        if ( v4 )
        {
          v67 = (0xC949D7C7509E6557LL * ((unsigned int)v4 + 4 * *(unsigned __int8 *)(a2 - 1)))
              ^ (0x9AE16A3B2F90404FLL * (*(unsigned __int8 *)a1 + (*((unsigned __int8 *)a1 + (v4 >> 1)) << 8)))
              ^ qword_4F99938;
          return 0x9AE16A3B2F90404FLL * (v67 ^ (v67 >> 47));
        }
      }
    }
    else
    {
      v8 = 0xB492B66FBE98F273LL * *a1;
      v9 = a1[1];
      v10 = 0x9AE16A3B2F90404FLL * *(_QWORD *)(a2 - 8);
      v11 = __ROR8__(v9 ^ 0xC949D7C7509E6557LL, 20) + qword_4F99938 + v8 + v4 - v10;
      v12 = 0xC3A5C85C97CB3127LL * *(_QWORD *)(a2 - 16) + __ROL8__(v8 - v9, 21);
      v13 = __ROR8__(v10 ^ qword_4F99938, 30);
      v14 = 0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v11 ^ (v12 + v13))) ^ v11 ^ ((0x9DDFEA08EB382D69LL * (v11 ^ (v12 + v13))) >> 47));
      return 0x9DDFEA08EB382D69LL * (v14 ^ (v14 >> 47));
    }
  }
  else
  {
    v5 = *(unsigned int *)(a2 - 4) ^ (unsigned __int64)qword_4F99938;
    v6 = 0x9DDFEA08EB382D69LL * (v5 ^ (v4 + 8LL * *(unsigned int *)a1));
    return 0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v6 ^ v5 ^ (v6 >> 47))) ^ ((0x9DDFEA08EB382D69LL * (v6 ^ v5 ^ (v6 >> 47))) >> 47));
  }
  return result;
}
