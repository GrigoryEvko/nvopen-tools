// Function: sub_1A73460
// Address: 0x1a73460
//
void __fastcall sub_1A73460(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 j; // r13
  __int64 v11; // rdi
  __int64 v12; // r12
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  int v18; // edi
  unsigned int v19; // r8d
  __int64 *v20; // rdx
  __int64 v21; // r10
  __int64 *v22; // rax
  unsigned int v23; // r8d
  __int64 *v24; // rdx
  __int64 v25; // r11
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r12
  __int64 v29; // r15
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rsi
  int v33; // edi
  unsigned int v34; // r9d
  __int64 *v35; // rdx
  __int64 v36; // r10
  __int64 *v37; // rax
  unsigned int v38; // r9d
  __int64 *v39; // rdx
  __int64 v40; // r10
  __int64 v41; // r14
  __int64 v42; // rax
  _BYTE *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  int v46; // r8d
  int v47; // r9d
  _BYTE *v48; // rsi
  unsigned __int64 v50; // r13
  _QWORD *v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rdi
  __int64 v54; // rcx
  int v55; // esi
  unsigned int v56; // r8d
  __int64 *v57; // rax
  __int64 v58; // r9
  __int64 *v59; // r8
  __int64 v60; // rbx
  unsigned int v61; // edi
  __int64 *v62; // rax
  __int64 v63; // r9
  __int64 v64; // r12
  __int64 v65; // rax
  _BYTE *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  int v69; // r8d
  int v70; // r9d
  _BYTE *v71; // rsi
  int v72; // edx
  int v73; // edx
  int v74; // r8d
  int v75; // edx
  int v76; // ecx
  int v77; // edx
  int v78; // eax
  int v79; // eax
  int v80; // r10d
  int v81; // r10d
  int v82; // r8d
  int v83; // ecx
  __int64 i; // [rsp+10h] [rbp-50h]
  __int64 v88[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *a2;
  if ( (*a2 & 4) != 0 )
  {
    v6 = *(_QWORD *)(a2[4] + 8);
    for ( i = a2[4]; v6; v6 = *(_QWORD *)(v6 + 8) )
    {
      if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v6) + 16) - 25) <= 9u )
        break;
    }
    v7 = v6;
    v8 = 0;
    v9 = a3;
    while ( v7 )
    {
      for ( j = *(_QWORD *)(v7 + 8); j; j = *(_QWORD *)(j + 8) )
      {
        if ( (unsigned __int8)(*((_BYTE *)sub_1648700(j) + 16) - 25) <= 9u )
          break;
      }
      v11 = v7;
      v7 = j;
      v12 = sub_1648700(v11)[5];
      if ( !(unsigned __int8)sub_1443560(a2, v12) )
        continue;
      sub_1A72700(a1, v12, i);
      v13 = sub_157EBA0(v12);
      sub_1648780(v13, i, v9);
      sub_1A73280(a1, v12, v9);
      if ( a4 )
      {
        if ( v8 )
        {
          v14 = *(_QWORD *)(*(_QWORD *)(v8 + 56) + 80LL);
          if ( v14 )
            v14 -= 24;
          if ( v8 != v14 && v12 != v14 )
          {
            v15 = *(_QWORD *)(a1 + 216);
            v16 = *(_QWORD *)(v15 + 32);
            v17 = *(unsigned int *)(v15 + 48);
            if ( !(_DWORD)v17 )
              goto LABEL_67;
            v18 = v17 - 1;
            v19 = (v17 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v20 = (__int64 *)(v16 + 16LL * v19);
            v21 = *v20;
            if ( v8 == *v20 )
            {
LABEL_17:
              v22 = (__int64 *)(v16 + 16 * v17);
              if ( v20 != v22 )
              {
                v8 = v20[1];
LABEL_19:
                v23 = v18 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
                v24 = (__int64 *)(v16 + 16LL * v23);
                v25 = *v24;
                if ( v12 == *v24 )
                {
LABEL_20:
                  if ( v22 != v24 )
                  {
                    v26 = v24[1];
                    if ( v8 )
                    {
                      if ( v26 )
                      {
                        while ( v26 != v8 )
                        {
                          if ( *(_DWORD *)(v8 + 16) < *(_DWORD *)(v26 + 16) )
                          {
                            v27 = v8;
                            v8 = v26;
                            v26 = v27;
                          }
                          v8 = *(_QWORD *)(v8 + 8);
                          if ( !v8 )
                            goto LABEL_27;
                        }
                        v8 = *(_QWORD *)v8;
                        v7 = j;
                        continue;
                      }
                    }
                  }
                }
                else
                {
                  v75 = 1;
                  while ( v25 != -8 )
                  {
                    v76 = v75 + 1;
                    v23 = v18 & (v75 + v23);
                    v24 = (__int64 *)(v16 + 16LL * v23);
                    v25 = *v24;
                    if ( v12 == *v24 )
                      goto LABEL_20;
                    v75 = v76;
                  }
                }
LABEL_67:
                v8 = 0;
                v7 = j;
                continue;
              }
            }
            else
            {
              v77 = 1;
              while ( v21 != -8 )
              {
                v83 = v77 + 1;
                v19 = v18 & (v77 + v19);
                v20 = (__int64 *)(v16 + 16LL * v19);
                v21 = *v20;
                if ( v8 == *v20 )
                  goto LABEL_17;
                v77 = v83;
              }
              v22 = (__int64 *)(v16 + 16 * v17);
            }
            v8 = 0;
            goto LABEL_19;
          }
          v8 = v14;
          v7 = j;
        }
        else
        {
          v8 = v12;
          v7 = j;
        }
      }
      else
      {
LABEL_27:
        v7 = j;
      }
    }
    v28 = v9;
    v29 = v7;
    if ( !v8 )
      goto LABEL_43;
    v30 = *(_QWORD *)(a1 + 216);
    v31 = *(unsigned int *)(v30 + 48);
    v32 = *(_QWORD *)(v30 + 32);
    if ( (_DWORD)v31 )
    {
      v33 = v31 - 1;
      v34 = (v31 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v35 = (__int64 *)(v32 + 16LL * v34);
      v36 = *v35;
      if ( v8 == *v35 )
      {
LABEL_32:
        v37 = (__int64 *)(v32 + 16 * v31);
        if ( v35 != v37 )
          v29 = v35[1];
      }
      else
      {
        v72 = 1;
        while ( v36 != -8 )
        {
          v82 = v72 + 1;
          v34 = v33 & (v72 + v34);
          v35 = (__int64 *)(v32 + 16LL * v34);
          v36 = *v35;
          if ( v8 == *v35 )
            goto LABEL_32;
          v72 = v82;
        }
        v37 = (__int64 *)(v32 + 16 * v31);
      }
      v38 = v33 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v39 = (__int64 *)(v32 + 16LL * v38);
      v40 = *v39;
      if ( v28 == *v39 )
      {
LABEL_35:
        if ( v37 != v39 )
        {
          v41 = v39[1];
          *(_BYTE *)(v30 + 72) = 0;
          v42 = *(_QWORD *)(v41 + 8);
          if ( v29 != v42 )
          {
            v88[0] = v41;
            v43 = sub_1A6CF50(*(_QWORD **)(v42 + 24), *(_QWORD *)(v42 + 32), v88);
            sub_15CDF70(*(_QWORD *)(v41 + 8) + 24LL, v43);
            *(_QWORD *)(v41 + 8) = v29;
            v88[0] = v41;
            v48 = *(_BYTE **)(v29 + 32);
            if ( v48 == *(_BYTE **)(v29 + 40) )
            {
              sub_15CE310(v29 + 24, v48, v88);
            }
            else
            {
              if ( v48 )
              {
                *(_QWORD *)v48 = v41;
                v48 = *(_BYTE **)(v29 + 32);
              }
              v48 += 8;
              *(_QWORD *)(v29 + 32) = v48;
            }
            if ( *(_DWORD *)(v41 + 16) != *(_DWORD *)(*(_QWORD *)(v41 + 8) + 16LL) + 1 )
              sub_1A6CC30(v41, (__int64)v48, v44, v45, v46, v47);
          }
LABEL_43:
          sub_1442F80((__int64)a2, v28);
          return;
        }
      }
      else
      {
        v73 = 1;
        while ( v40 != -8 )
        {
          v74 = v73 + 1;
          v38 = v33 & (v73 + v38);
          v39 = (__int64 *)(v32 + 16LL * v38);
          v40 = *v39;
          if ( v28 == *v39 )
            goto LABEL_35;
          v73 = v74;
        }
      }
    }
    *(_BYTE *)(v30 + 72) = 0;
    BUG();
  }
  v50 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  sub_1A72E50(a1, v5 & 0xFFFFFFFFFFFFFFF8LL);
  v51 = sub_1648A60(56, 1u);
  if ( v51 )
    sub_15F8590((__int64)v51, a3, v50);
  sub_1A73280(a1, v50, a3);
  if ( a4 )
  {
    v52 = *(_QWORD *)(a1 + 216);
    v53 = *(unsigned int *)(v52 + 48);
    v54 = *(_QWORD *)(v52 + 32);
    if ( !(_DWORD)v53 )
      goto LABEL_103;
    v55 = v53 - 1;
    v56 = (v53 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
    v57 = (__int64 *)(v54 + 16LL * v56);
    v58 = *v57;
    if ( v50 == *v57 )
    {
LABEL_56:
      v59 = (__int64 *)(v54 + 16 * v53);
      if ( v57 != v59 )
      {
        v60 = v57[1];
        goto LABEL_58;
      }
    }
    else
    {
      v78 = 1;
      while ( v58 != -8 )
      {
        v81 = v78 + 1;
        v56 = v55 & (v78 + v56);
        v57 = (__int64 *)(v54 + 16LL * v56);
        v58 = *v57;
        if ( v50 == *v57 )
          goto LABEL_56;
        v78 = v81;
      }
      v59 = (__int64 *)(v54 + 16LL * (unsigned int)v53);
    }
    v60 = 0;
LABEL_58:
    v61 = v55 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v62 = (__int64 *)(v54 + 16LL * v61);
    v63 = *v62;
    if ( a3 == *v62 )
    {
LABEL_59:
      if ( v59 != v62 )
      {
        v64 = v62[1];
        *(_BYTE *)(v52 + 72) = 0;
        v65 = *(_QWORD *)(v64 + 8);
        if ( v60 != v65 )
        {
          v88[0] = v64;
          v66 = sub_1A6CF50(*(_QWORD **)(v65 + 24), *(_QWORD *)(v65 + 32), v88);
          sub_15CDF70(*(_QWORD *)(v64 + 8) + 24LL, v66);
          *(_QWORD *)(v64 + 8) = v60;
          v88[0] = v64;
          v71 = *(_BYTE **)(v60 + 32);
          if ( v71 == *(_BYTE **)(v60 + 40) )
          {
            sub_15CE310(v60 + 24, v71, v88);
          }
          else
          {
            if ( v71 )
            {
              *(_QWORD *)v71 = v64;
              v71 = *(_BYTE **)(v60 + 32);
            }
            v71 += 8;
            *(_QWORD *)(v60 + 32) = v71;
          }
          if ( *(_DWORD *)(v64 + 16) != *(_DWORD *)(*(_QWORD *)(v64 + 8) + 16LL) + 1 )
            sub_1A6CC30(v64, (__int64)v71, v67, v68, v69, v70);
        }
        return;
      }
    }
    else
    {
      v79 = 1;
      while ( v63 != -8 )
      {
        v80 = v79 + 1;
        v61 = v55 & (v79 + v61);
        v62 = (__int64 *)(v54 + 16LL * v61);
        v63 = *v62;
        if ( a3 == *v62 )
          goto LABEL_59;
        v79 = v80;
      }
    }
LABEL_103:
    *(_BYTE *)(v52 + 72) = 0;
    BUG();
  }
}
