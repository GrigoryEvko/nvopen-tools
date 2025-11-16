// Function: sub_1E30AD0
// Address: 0x1e30ad0
//
unsigned __int64 __fastcall sub_1E30AD0(char *src, __int64 a2, char *a3, char *a4)
{
  __int64 v6; // rcx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r10
  __int64 v15; // r8
  __int64 v16; // rcx
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // r14
  char *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdi
  char *v24; // rdx
  char *v25; // rdi
  char v26; // r8
  char v27; // si
  __int64 v28; // rdx
  char *v29; // rsi
  char *v30; // rdx
  char v31; // cl
  char v32; // r8
  char *v33; // rax
  char v34; // si
  char v35; // cl
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // r11
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r9
  __int64 v44; // rsi
  __int64 v45; // rax
  unsigned __int64 v46; // r8
  unsigned __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rdi
  __int64 v50; // r11
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // r9
  unsigned __int64 v54; // r11
  unsigned __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rdi
  __int64 v61; // rbx
  __int64 v62; // rcx
  __int64 v63; // rsi
  unsigned __int64 v64; // rsi
  unsigned __int64 v65; // rax
  unsigned __int64 v66; // rax
  __int64 v67; // rdi
  __int64 v68; // rdx
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rsi
  __int64 v71; // r10
  __int64 v72; // r11
  __int64 v73; // rax
  __int64 v74; // rcx
  __int64 v75; // rax
  __int64 v76; // rbx
  __int64 v77; // r8
  __int64 v78; // rcx
  __int64 v79; // rax
  unsigned __int64 v80; // rdx
  unsigned __int64 v81; // rdx
  size_t v82; // rax
  char v83; // r15
  size_t v84; // rdx
  char v85; // r15
  size_t v86; // rax
  size_t v87; // [rsp+8h] [rbp-38h]

  v6 = a3 - src;
  if ( a2 )
  {
    v20 = a3 - src;
    if ( a3 != src && a3 != a4 )
    {
      v21 = src;
      v22 = a4 - src;
      if ( a4 - a3 == v6 )
      {
        v33 = a3;
        do
        {
          v34 = *v33;
          v35 = *v21++;
          ++v33;
          *(v21 - 1) = v34;
          *(v33 - 1) = v35;
        }
        while ( a3 != v21 );
      }
      else
      {
        while ( 1 )
        {
          v23 = v22 - v6;
          if ( v6 < v22 - v6 )
            break;
LABEL_18:
          if ( v23 == 1 )
          {
            v85 = v21[v22 - 1];
            v86 = v22 - 1;
            if ( v86 )
              memmove(v21 + 1, v21, v86);
            *v21 = v85;
            goto LABEL_26;
          }
          v29 = &v21[v22];
          v30 = &v21[v22 - v23];
          v21 = v30;
          if ( v6 > 0 )
          {
            v21 = &v30[-v6];
            do
            {
              v31 = *(v30 - 1);
              v32 = *(v29 - 1);
              --v30;
              --v29;
              *v30 = v32;
              *v29 = v31;
            }
            while ( v30 != v21 );
          }
          v6 = v22 % v23;
          if ( !(v22 % v23) )
            goto LABEL_26;
          v22 = v23;
        }
        while ( v6 != 1 )
        {
          v24 = &v21[v6];
          if ( v23 <= 0 )
          {
            v25 = v21;
          }
          else
          {
            v25 = &v21[v23];
            do
            {
              v26 = *v24;
              v27 = *v21++;
              ++v24;
              *(v21 - 1) = v26;
              *(v24 - 1) = v27;
            }
            while ( v21 != v25 );
          }
          v28 = v22 % v6;
          if ( !(v22 % v6) )
            goto LABEL_26;
          v22 = v6;
          v21 = v25;
          v6 -= v28;
          v23 = v22 - v6;
          if ( v6 >= v22 - v6 )
            goto LABEL_18;
        }
        v82 = v22 - 1;
        v83 = *v21;
        v84 = v82;
        if ( v82 )
        {
          v87 = v82;
          memmove(v21, v21 + 1, v82);
          v84 = v87;
        }
        v21[v84] = v83;
      }
    }
LABEL_26:
    v36 = *((_QWORD *)src + 9);
    v37 = *((_QWORD *)src + 12);
    v38 = *((_QWORD *)src + 14);
    v39 = __ROL8__(*((_QWORD *)src + 1) + *((_QWORD *)src + 11) + v36 + *((_QWORD *)src + 8), 27);
    v40 = *((_QWORD *)src + 10);
    v41 = __ROL8__(*((_QWORD *)src + 6) + v36 + v37, 22);
    v42 = *(_QWORD *)src - 0x4B6D499041670D8DLL * v37;
    v43 = v42 + *((_QWORD *)src + 2) + *((_QWORD *)src + 1);
    v44 = v42;
    v45 = *((_QWORD *)src + 3);
    v46 = v38 ^ (0xB492B66FBE98F273LL * v39);
    v47 = *((_QWORD *)src + 5) + *((_QWORD *)src + 11) - 0x4B6D499041670D8DLL * v41;
    v48 = *((_QWORD *)src + 13);
    v49 = *((_QWORD *)src + 4) + v38;
    *((_QWORD *)src + 10) = v46;
    *((_QWORD *)src + 9) = v47;
    v50 = v48 + v40;
    v51 = v48 + v45;
    v52 = v43;
    v53 = *((_QWORD *)src + 3) + v43;
    *((_QWORD *)src + 11) = v53;
    v54 = 0xB492B66FBE98F273LL * __ROL8__(v50, 31);
    v55 = v46 + v44 + v51;
    v56 = __ROL8__(v52, 20) + v44;
    v57 = *((_QWORD *)src + 5) + *((_QWORD *)src + 6);
    v58 = v56 + __ROR8__(v55, 21);
    v59 = *((_QWORD *)src + 2) + *((_QWORD *)src + 7);
    v60 = v54 + v49;
    *((_QWORD *)src + 12) = v58;
    *((_QWORD *)src + 8) = v54;
    v61 = v60 + v57;
    v62 = *((_QWORD *)src + 7) + v60 + v57;
    *((_QWORD *)src + 13) = v62;
    v63 = __ROL8__(v61, 20) + v60 + __ROR8__(v47 + v60 + v59, 21);
    *((_QWORD *)src + 14) = v63;
    v64 = 0xB492B66FBE98F273LL * (((unsigned __int64)(v20 + a2) >> 47) ^ (v20 + a2))
        + v54
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v63 ^ v58)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v63 ^ v58)) ^ v63)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v63 ^ v58)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v63 ^ v58)) ^ v63)));
    v65 = ((0x9DDFEA08EB382D69LL
          * (v62 ^ (0x9DDFEA08EB382D69LL * (v62 ^ v53)) ^ ((0x9DDFEA08EB382D69LL * (v62 ^ v53)) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v62 ^ (0x9DDFEA08EB382D69LL * (v62 ^ v53)) ^ ((0x9DDFEA08EB382D69LL * (v62 ^ v53)) >> 47)));
    v66 = 0x9DDFEA08EB382D69LL
        * (v64
         ^ (0x9DDFEA08EB382D69LL
          * (v64 ^ (v46 - 0x4B6D499041670D8DLL * (v47 ^ (v47 >> 47)) - 0x622015F714C7D297LL * v65)))
         ^ ((0x9DDFEA08EB382D69LL
           * (v64 ^ (v46 - 0x4B6D499041670D8DLL * (v47 ^ (v47 >> 47)) - 0x622015F714C7D297LL * v65))) >> 47));
    return 0x9DDFEA08EB382D69LL * ((v66 >> 47) ^ v66);
  }
  else
  {
    v8 = *((_QWORD *)src + 15);
    if ( (unsigned __int64)(v6 - 4) > 4 )
    {
      if ( (unsigned __int64)(v6 - 9) <= 7 )
      {
        v67 = *((_QWORD *)a3 - 1);
        v68 = __ROR8__(v67 + v6, v6);
        v69 = 0x9DDFEA08EB382D69LL
            * ((0x9DDFEA08EB382D69LL * (v68 ^ *(_QWORD *)src ^ v8))
             ^ v68
             ^ ((0x9DDFEA08EB382D69LL * (v68 ^ *(_QWORD *)src ^ v8)) >> 47));
        return v67 ^ (0x9DDFEA08EB382D69LL * (v69 ^ (v69 >> 47)));
      }
      else if ( (unsigned __int64)(v6 - 17) > 0xF )
      {
        if ( (unsigned __int64)v6 > 0x20 )
        {
          v71 = *((_QWORD *)a3 - 2);
          v72 = *((_QWORD *)src + 3);
          v73 = v71 + v6;
          v74 = *((_QWORD *)src + 2);
          v75 = *(_QWORD *)src - 0x3C5A37A36834CED9LL * v73;
          v76 = v75 + *((_QWORD *)src + 1);
          v77 = v76 + v74;
          v78 = *((_QWORD *)a3 - 4) + v74;
          v79 = __ROR8__(v77, 31) + __ROR8__(v76, 7) + __ROL8__(v72 + v75, 12) + __ROL8__(v75, 27);
          v80 = 0xC3A5C85C97CB3127LL * (v79 + *((_QWORD *)a3 - 1) + v78 + *((_QWORD *)a3 - 3) + v71)
              - 0x651E95C4D06FBFB1LL
              * (v77
               + __ROR8__(v78 + *((_QWORD *)a3 - 3) + v71, 31)
               + __ROR8__(v78 + *((_QWORD *)a3 - 3), 7)
               + v72
               + __ROL8__(v78, 27)
               + __ROL8__(v78 + *((_QWORD *)a3 - 1), 12));
          v81 = ((0xC3A5C85C97CB3127LL * ((v80 >> 47) ^ v80)) ^ v8) + v79;
          return 0x9AE16A3B2F90404FLL * (v81 ^ (v81 >> 47));
        }
        else
        {
          result = v8 ^ 0x9AE16A3B2F90404FLL;
          if ( v6 )
          {
            v70 = (0xC949D7C7509E6557LL * ((unsigned int)v6 + 4 * (unsigned __int8)*(a3 - 1)))
                ^ (0x9AE16A3B2F90404FLL
                 * ((unsigned __int8)*src + ((unsigned __int8)src[(unsigned __int64)v6 >> 1] << 8)))
                ^ v8;
            return 0x9AE16A3B2F90404FLL * (v70 ^ (v70 >> 47));
          }
        }
      }
      else
      {
        v11 = 0xB492B66FBE98F273LL * *(_QWORD *)src;
        v12 = *((_QWORD *)src + 1);
        v13 = 0x9AE16A3B2F90404FLL * *((_QWORD *)a3 - 1);
        v14 = v8 + v11;
        v15 = __ROL8__(v11 - v12, 21);
        v16 = __ROR8__(v12 ^ 0xC949D7C7509E6557LL, 20) + v14 + v6;
        v17 = 0xC3A5C85C97CB3127LL * *((_QWORD *)a3 - 2);
        v18 = __ROR8__(v13 ^ v8, 30);
        v19 = 0x9DDFEA08EB382D69LL
            * ((v16 - v13)
             ^ (0x9DDFEA08EB382D69LL * ((v16 - v13) ^ (v15 + v17 + v18)))
             ^ ((0x9DDFEA08EB382D69LL * ((v16 - v13) ^ (v15 + v17 + v18))) >> 47));
        return 0x9DDFEA08EB382D69LL * ((v19 >> 47) ^ v19);
      }
    }
    else
    {
      v9 = 0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (*((unsigned int *)a3 - 1) ^ v8 ^ (v6 + 8LL * *(unsigned int *)src)))
          ^ *((unsigned int *)a3 - 1)
          ^ v8
          ^ ((0x9DDFEA08EB382D69LL * (*((unsigned int *)a3 - 1) ^ v8 ^ (v6 + 8LL * *(unsigned int *)src))) >> 47));
      return 0x9DDFEA08EB382D69LL * (v9 ^ (v9 >> 47));
    }
  }
  return result;
}
