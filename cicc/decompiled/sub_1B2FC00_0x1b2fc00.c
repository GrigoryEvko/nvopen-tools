// Function: sub_1B2FC00
// Address: 0x1b2fc00
//
__int64 __fastcall sub_1B2FC00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v5; // dl
  __int64 *v6; // rax
  __int64 result; // rax
  char *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // r15
  char v12; // cl
  int v13; // esi
  _QWORD *v14; // rdi
  unsigned int v15; // edx
  _QWORD *v16; // rax
  __int64 v17; // r8
  int v18; // edx
  unsigned int v19; // esi
  char v20; // r10
  unsigned int v21; // edx
  unsigned int v22; // edi
  unsigned int v23; // r8d
  __int64 v24; // r11
  __int64 v25; // r15
  __int64 v26; // r13
  char v27; // cl
  __int64 v28; // rsi
  _QWORD *v29; // rdi
  int v30; // edx
  __int64 v31; // rbx
  unsigned int v32; // r12d
  unsigned int v33; // r9d
  __int64 v34; // r8
  __int64 v35; // rcx
  __int64 *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // r11d
  int v40; // r11d
  _QWORD *v41; // r9
  _QWORD *v42; // rsi
  int v43; // edi
  unsigned int v44; // ecx
  __int64 v45; // r8
  int v46; // r9d
  _QWORD *v47; // rdx
  int v48; // esi
  _QWORD *v49; // rdi
  unsigned int v50; // ecx
  __int64 v51; // r8
  int v52; // r9d
  unsigned int v53; // esi
  __int64 v54; // r11
  unsigned __int64 v55; // rcx
  unsigned __int64 v56; // rcx
  unsigned int i; // ecx
  __int64 *v58; // rdx
  __int64 v59; // rdi
  unsigned int v60; // ecx
  int v61; // ecx
  _QWORD *v62; // r10
  int v63; // esi
  __int64 v64; // r9
  int v65; // esi
  __int64 *v66; // rax
  int v67; // r8d
  unsigned __int64 v68; // rcx
  unsigned __int64 v69; // rcx
  unsigned int k; // ecx
  __int64 v71; // rdi
  unsigned int v72; // ecx
  int v73; // esi
  __int64 v74; // r9
  int v75; // esi
  int v76; // r8d
  unsigned int j; // ecx
  __int64 v78; // rdi
  unsigned int v79; // ecx
  unsigned __int64 v80; // [rsp+8h] [rbp-188h]
  __int64 v82; // [rsp+18h] [rbp-178h]
  unsigned __int64 v83; // [rsp+18h] [rbp-178h]
  __int64 v84; // [rsp+20h] [rbp-170h]
  int v85; // [rsp+20h] [rbp-170h]
  __int64 v88; // [rsp+40h] [rbp-150h]
  __int64 *v89; // [rsp+48h] [rbp-148h]
  __int64 v90; // [rsp+50h] [rbp-140h] BYREF
  __int64 v91; // [rsp+58h] [rbp-138h]
  _QWORD *v92; // [rsp+60h] [rbp-130h] BYREF
  unsigned int v93; // [rsp+68h] [rbp-128h]
  char v94; // [rsp+160h] [rbp-30h] BYREF

  v5 = *(_BYTE *)(a2 + 23);
  if ( (v5 & 0x40) != 0 )
    v6 = *(__int64 **)(a2 - 8);
  else
    v6 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v88 = *v6;
  result = *(unsigned __int8 *)(*v6 + 16);
  if ( (_BYTE)result != 17 && (unsigned __int8)result <= 0x17u )
    return result;
  result = *(_QWORD *)(v88 + 8);
  if ( result )
  {
    if ( !*(_QWORD *)(result + 8) )
      return result;
  }
  v8 = (char *)&v92;
  v90 = 0;
  v91 = 1;
  do
  {
    *(_QWORD *)v8 = -8;
    v8 += 16;
  }
  while ( v8 != &v94 );
  if ( !((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) )
  {
    v24 = 0xFFFFFFFFLL;
    goto LABEL_35;
  }
  v9 = 24;
  v10 = 48LL * (((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1) + 72;
LABEL_17:
  if ( (v5 & 0x40) == 0 )
  {
    v11 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) + v9);
    v12 = v91 & 1;
    if ( (v91 & 1) == 0 )
      goto LABEL_19;
LABEL_12:
    v13 = 15;
    v14 = &v92;
    goto LABEL_13;
  }
  v11 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + v9);
  v12 = v91 & 1;
  if ( (v91 & 1) != 0 )
    goto LABEL_12;
LABEL_19:
  v19 = v93;
  v14 = v92;
  if ( !v93 )
  {
    v21 = v91;
    ++v90;
    v16 = 0;
    v22 = ((unsigned int)v91 >> 1) + 1;
LABEL_27:
    v23 = 3 * v19;
    goto LABEL_28;
  }
  v13 = v93 - 1;
LABEL_13:
  v15 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v16 = &v14[2 * v15];
  v17 = *v16;
  if ( v11 == *v16 )
  {
    v18 = *((_DWORD *)v16 + 2) + 1;
    goto LABEL_15;
  }
  v40 = 1;
  v41 = 0;
  while ( v17 != -8 )
  {
    if ( v41 || v17 != -16 )
      v16 = v41;
    v15 = v13 & (v40 + v15);
    v62 = &v14[2 * v15];
    v17 = *v62;
    if ( v11 == *v62 )
    {
      v18 = *((_DWORD *)v62 + 2) + 1;
      v16 = v62;
LABEL_15:
      v9 += 48;
      *((_DWORD *)v16 + 2) = v18;
      if ( v10 == v9 )
        goto LABEL_33;
LABEL_16:
      v5 = *(_BYTE *)(a2 + 23);
      goto LABEL_17;
    }
    ++v40;
    v41 = v16;
    v16 = &v14[2 * v15];
  }
  v21 = v91;
  v23 = 48;
  v19 = 16;
  if ( v41 )
    v16 = v41;
  ++v90;
  v22 = ((unsigned int)v91 >> 1) + 1;
  if ( !v12 )
  {
    v19 = v93;
    goto LABEL_27;
  }
LABEL_28:
  if ( 4 * v22 >= v23 )
  {
    sub_1917CA0((__int64)&v90, 2 * v19);
    if ( (v91 & 1) != 0 )
    {
      v43 = 15;
      v42 = &v92;
    }
    else
    {
      v42 = v92;
      if ( !v93 )
        goto LABEL_151;
      v43 = v93 - 1;
    }
    v21 = v91;
    v44 = v43 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v16 = &v42[2 * v44];
    v45 = *v16;
    if ( v11 == *v16 )
      goto LABEL_30;
    v46 = 1;
    v47 = 0;
    while ( v45 != -8 )
    {
      if ( v45 == -16 && !v47 )
        v47 = v16;
      v44 = v43 & (v46 + v44);
      v16 = &v42[2 * v44];
      v45 = *v16;
      if ( v11 == *v16 )
        goto LABEL_73;
      ++v46;
    }
LABEL_71:
    if ( v47 )
      v16 = v47;
LABEL_73:
    v21 = v91;
    goto LABEL_30;
  }
  if ( v19 - HIDWORD(v91) - v22 <= v19 >> 3 )
  {
    sub_1917CA0((__int64)&v90, v19);
    if ( (v91 & 1) != 0 )
    {
      v48 = 15;
      v49 = &v92;
LABEL_78:
      v21 = v91;
      v50 = v48 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v16 = &v49[2 * v50];
      v51 = *v16;
      if ( v11 == *v16 )
        goto LABEL_30;
      v52 = 1;
      v47 = 0;
      while ( v51 != -8 )
      {
        if ( v51 == -16 && !v47 )
          v47 = v16;
        v50 = v48 & (v52 + v50);
        v16 = &v49[2 * v50];
        v51 = *v16;
        if ( v11 == *v16 )
          goto LABEL_73;
        ++v52;
      }
      goto LABEL_71;
    }
    v49 = v92;
    if ( v93 )
    {
      v48 = v93 - 1;
      goto LABEL_78;
    }
LABEL_151:
    LODWORD(v91) = (2 * ((unsigned int)v91 >> 1) + 2) | v91 & 1;
    BUG();
  }
LABEL_30:
  LODWORD(v91) = (2 * (v21 >> 1) + 2) | v21 & 1;
  if ( *v16 != -8 )
    --HIDWORD(v91);
  v9 += 48;
  *((_DWORD *)v16 + 2) = 0;
  *v16 = v11;
  *((_DWORD *)v16 + 2) = 1;
  if ( v10 != v9 )
    goto LABEL_16;
LABEL_33:
  result = (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1;
  v24 = (unsigned int)(result - 1);
  if ( (_DWORD)result != 1 )
  {
    v5 = *(_BYTE *)(a2 + 23);
LABEL_35:
    v25 = 0;
    v26 = v24;
    v27 = v91;
    v80 = (unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32;
    while ( 1 )
    {
      v38 = 24;
      if ( (_DWORD)v25 != -2 )
        v38 = 24LL * (unsigned int)(2 * v25 + 3);
      if ( (v5 & 0x40) != 0 )
      {
        v28 = *(_QWORD *)(a2 - 8);
        ++v25;
        result = v28 + v38;
        v20 = v27 & 1;
        if ( (v27 & 1) != 0 )
          goto LABEL_37;
      }
      else
      {
        ++v25;
        v28 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        result = v28 + v38;
        v20 = v27 & 1;
        if ( (v27 & 1) != 0 )
        {
LABEL_37:
          v29 = &v92;
          v30 = 15;
          goto LABEL_38;
        }
      }
      v29 = v92;
      if ( !v93 )
        goto LABEL_46;
      v30 = v93 - 1;
LABEL_38:
      v31 = *(_QWORD *)result;
      v32 = ((unsigned int)*(_QWORD *)result >> 9) ^ ((unsigned int)*(_QWORD *)result >> 4);
      v33 = v30 & v32;
      result = (__int64)&v29[2 * (v30 & v32)];
      v34 = *(_QWORD *)result;
      if ( *(_QWORD *)result == v31 )
      {
LABEL_39:
        if ( *(_DWORD *)(result + 8) == 1 )
        {
          v82 = *(_QWORD *)(a2 + 40);
          v84 = *(_QWORD *)(v28 + 24LL * (unsigned int)(2 * v25));
          v35 = sub_22077B0(80);
          if ( v35 )
          {
            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
              v36 = *(__int64 **)(a2 - 8);
            else
              v36 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
            v37 = *v36;
            *(_QWORD *)(v35 + 48) = v82;
            *(_QWORD *)(v35 + 8) = 0;
            *(_QWORD *)(v35 + 40) = v37;
            *(_QWORD *)(v35 + 16) = 0;
            *(_DWORD *)(v35 + 24) = 2;
            *(_QWORD *)(v35 + 32) = v88;
            *(_QWORD *)(v35 + 56) = v31;
            *(_QWORD *)v35 = &unk_49F6760;
            *(_QWORD *)(v35 + 64) = v84;
            *(_QWORD *)(v35 + 72) = a2;
          }
          sub_1B2EEE0((__int64 *)a1, a4, v88, v35);
          result = sub_157F0B0(v31);
          if ( !result )
          {
            v53 = *(_DWORD *)(a1 + 3256);
            v54 = a1 + 3232;
            if ( v53 )
            {
              v85 = 1;
              v89 = 0;
              v55 = ((v32 | v80) - 1 - ((unsigned __int64)v32 << 32))
                  ^ (((v32 | v80) - 1 - ((unsigned __int64)v32 << 32)) >> 22);
              v56 = 9 * (((v55 - 1 - (v55 << 13)) >> 8) ^ (v55 - 1 - (v55 << 13)));
              v83 = (((v56 ^ (v56 >> 15)) - 1 - ((v56 ^ (v56 >> 15)) << 27)) >> 31)
                  ^ ((v56 ^ (v56 >> 15)) - 1 - ((v56 ^ (v56 >> 15)) << 27));
              for ( i = (v53 - 1)
                      & ((((v56 ^ (v56 >> 15)) - 1 - ((v56 ^ (v56 >> 15)) << 27)) >> 31)
                       ^ ((v56 ^ (v56 >> 15)) - 1 - (((unsigned int)v56 ^ (unsigned int)(v56 >> 15)) << 27)));
                    ;
                    i = (v53 - 1) & v60 )
              {
                v58 = (__int64 *)(*(_QWORD *)(a1 + 3240) + 16LL * i);
                v59 = *v58;
                if ( a3 == *v58 && v31 == v58[1] )
                  goto LABEL_45;
                if ( v59 == -8 )
                {
                  if ( v58[1] == -8 )
                  {
                    if ( v89 )
                      v58 = v89;
                    ++*(_QWORD *)(a1 + 3232);
                    v61 = *(_DWORD *)(a1 + 3248) + 1;
                    if ( 4 * v61 < 3 * v53 )
                    {
                      if ( v53 - *(_DWORD *)(a1 + 3252) - v61 > v53 >> 3 )
                      {
LABEL_103:
                        *(_DWORD *)(a1 + 3248) = v61;
                        if ( *v58 != -8 || v58[1] != -8 )
                          --*(_DWORD *)(a1 + 3252);
                        result = a3;
                        v58[1] = v31;
                        *v58 = a3;
                        goto LABEL_45;
                      }
                      sub_1A15100(v54, v53);
                      v73 = *(_DWORD *)(a1 + 3256);
                      if ( v73 )
                      {
                        v74 = *(_QWORD *)(a1 + 3240);
                        v75 = v73 - 1;
                        v66 = 0;
                        v76 = 1;
                        for ( j = v75 & v83; ; j = v75 & v79 )
                        {
                          v58 = (__int64 *)(v74 + 16LL * j);
                          v78 = *v58;
                          if ( a3 == *v58 && v31 == v58[1] )
                            break;
                          if ( v78 == -8 )
                          {
                            if ( v58[1] == -8 )
                              goto LABEL_145;
                          }
                          else if ( v78 == -16 && v58[1] == -16 && !v66 )
                          {
                            v66 = (__int64 *)(v74 + 16LL * j);
                          }
                          v79 = v76 + j;
                          ++v76;
                        }
LABEL_141:
                        v61 = *(_DWORD *)(a1 + 3248) + 1;
                        goto LABEL_103;
                      }
                      goto LABEL_150;
                    }
LABEL_116:
                    sub_1A15100(v54, 2 * v53);
                    v63 = *(_DWORD *)(a1 + 3256);
                    if ( v63 )
                    {
                      v64 = *(_QWORD *)(a1 + 3240);
                      v65 = v63 - 1;
                      v66 = 0;
                      v67 = 1;
                      v68 = (((v32 | v80) - 1 - ((unsigned __int64)v32 << 32)) >> 22)
                          ^ ((v32 | v80) - 1 - ((unsigned __int64)v32 << 32));
                      v69 = ((9 * (((v68 - 1 - (v68 << 13)) >> 8) ^ (v68 - 1 - (v68 << 13)))) >> 15)
                          ^ (9 * (((v68 - 1 - (v68 << 13)) >> 8) ^ (v68 - 1 - (v68 << 13))));
                      for ( k = v65 & ((v69 - 1 - ((_DWORD)v69 << 27)) ^ ((v69 - 1 - (v69 << 27)) >> 31)); ; k = v65 & v72 )
                      {
                        v58 = (__int64 *)(v64 + 16LL * k);
                        v71 = *v58;
                        if ( a3 == *v58 && v31 == v58[1] )
                          break;
                        if ( v71 == -8 )
                        {
                          if ( v58[1] == -8 )
                          {
LABEL_145:
                            if ( v66 )
                              v58 = v66;
                            v61 = *(_DWORD *)(a1 + 3248) + 1;
                            goto LABEL_103;
                          }
                        }
                        else if ( v71 == -16 && v58[1] == -16 && !v66 )
                        {
                          v66 = (__int64 *)(v64 + 16LL * k);
                        }
                        v72 = v67 + k;
                        ++v67;
                      }
                      goto LABEL_141;
                    }
LABEL_150:
                    ++*(_DWORD *)(a1 + 3248);
                    BUG();
                  }
                }
                else if ( v59 == -16 && v58[1] == -16 )
                {
                  if ( v89 )
                    v58 = v89;
                  v89 = v58;
                }
                v60 = v85 + i;
                ++v85;
              }
            }
            ++*(_QWORD *)(a1 + 3232);
            goto LABEL_116;
          }
LABEL_45:
          v27 = v91;
          v20 = v91 & 1;
        }
      }
      else
      {
        result = 1;
        while ( v34 != -8 )
        {
          v39 = result + 1;
          v33 = v30 & (result + v33);
          result = (__int64)&v29[2 * v33];
          v34 = *(_QWORD *)result;
          if ( v31 == *(_QWORD *)result )
            goto LABEL_39;
          result = v39;
        }
      }
LABEL_46:
      if ( v26 == v25 )
        goto LABEL_22;
      v5 = *(_BYTE *)(a2 + 23);
    }
  }
  v20 = v91 & 1;
LABEL_22:
  if ( !v20 )
    return j___libc_free_0(v92);
  return result;
}
