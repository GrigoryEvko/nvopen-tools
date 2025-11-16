// Function: sub_291EF30
// Address: 0x291ef30
//
__int64 __fastcall sub_291EF30(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  unsigned __int8 *v4; // rax
  int v5; // edx
  __int64 v6; // r9
  unsigned __int8 **v7; // rax
  unsigned __int8 *v8; // r15
  unsigned __int8 v9; // cl
  __int64 result; // rax
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // rbx
  unsigned __int8 **v19; // r14
  unsigned __int8 **v20; // r13
  unsigned __int8 *v21; // rbx
  __int64 v22; // r15
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int8 *v25; // r12
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int8 **v28; // r14
  unsigned __int8 **v29; // r13
  unsigned __int8 *v30; // rbx
  __int64 v31; // r15
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  unsigned __int8 *v34; // r12
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned int v37; // r14d
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // r14
  int v41; // eax
  __int64 v42; // rdx
  int v43; // eax
  __int64 v44; // rdx
  _BYTE *v45; // rax
  unsigned int **v46; // r14
  __int64 v47; // rdx
  __int64 v48; // r14
  __int64 *v49; // rdx
  unsigned __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  int v53; // r9d
  __int64 v54; // [rsp+10h] [rbp-110h]
  __int64 v55; // [rsp+10h] [rbp-110h]
  unsigned int **v56; // [rsp+18h] [rbp-108h]
  unsigned int **v57; // [rsp+18h] [rbp-108h]
  __int64 v58; // [rsp+30h] [rbp-F0h]
  __int64 v59; // [rsp+30h] [rbp-F0h]
  _BYTE *v60; // [rsp+30h] [rbp-F0h]
  __int64 v61; // [rsp+38h] [rbp-E8h]
  __int64 *v62; // [rsp+38h] [rbp-E8h]
  __int64 *v63; // [rsp+38h] [rbp-E8h]
  __int64 v64; // [rsp+38h] [rbp-E8h]
  const char *v65; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v66; // [rsp+48h] [rbp-D8h]
  char *v67; // [rsp+50h] [rbp-D0h]
  __int16 v68; // [rsp+60h] [rbp-C0h]
  _BYTE *v69; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v70; // [rsp+78h] [rbp-A8h]
  _BYTE v71[48]; // [rsp+80h] [rbp-A0h] BYREF
  _BYTE *v72; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v73; // [rsp+B8h] [rbp-68h]
  _BYTE v74[96]; // [rsp+C0h] [rbp-60h] BYREF

  v2 = a1;
  v3 = (__int64)a2;
  v4 = sub_BD3990(*(unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)], (__int64)a2);
  if ( *v4 != 86 )
  {
    v5 = *((_DWORD *)a2 + 1);
    v6 = v5 & 0x7FFFFFF;
    v7 = (unsigned __int8 **)&a2[32 * (1 - v6)];
    if ( a2 != (unsigned __int8 *)v7 )
    {
      v8 = 0;
      goto LABEL_4;
    }
    return 0;
  }
  v5 = *((_DWORD *)a2 + 1);
  v8 = v4;
  v6 = v5 & 0x7FFFFFF;
  v7 = (unsigned __int8 **)&a2[32 * (1 - v6)];
  if ( a2 == (unsigned __int8 *)v7 )
  {
LABEL_13:
    v11 = 32 * v6;
    v58 = *((_QWORD *)v8 - 8);
    v12 = *((_QWORD *)v8 - 4);
    v70 = 0x600000000LL;
    v61 = v12;
    v69 = v71;
    if ( (*(_BYTE *)(v3 + 7) & 0x40) != 0 )
    {
      v13 = *(_QWORD *)(v3 - 8);
      v14 = v13 + v11;
      v15 = v13;
      if ( v13 == v13 + v11 )
      {
        v73 = 0x600000000LL;
        v72 = v74;
        v16 = 32LL * (v5 & 0x7FFFFFF);
LABEL_16:
        v17 = v13;
        v18 = v13 + v16;
LABEL_17:
        v19 = (unsigned __int8 **)v17;
        if ( v17 != v18 )
        {
          v54 = v2;
          v20 = (unsigned __int8 **)v18;
          v21 = v8;
          v22 = v3;
          do
          {
            v25 = *v19;
            if ( v21 == *v19
              || v25 == *(unsigned __int8 **)(v22 - 32LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF))
              && v21 == sub_BD3990(*v19, (__int64)a2) )
            {
              v26 = (unsigned int)v73;
              v27 = (unsigned int)v73 + 1LL;
              if ( v27 > HIDWORD(v73) )
              {
                a2 = v74;
                sub_C8D5F0((__int64)&v72, v74, v27, 8u, v17, v16);
                v26 = (unsigned int)v73;
              }
              *(_QWORD *)&v72[8 * v26] = v61;
              LODWORD(v73) = v73 + 1;
            }
            else
            {
              v23 = (unsigned int)v73;
              v24 = (unsigned int)v73 + 1LL;
              if ( v24 > HIDWORD(v73) )
              {
                a2 = v74;
                sub_C8D5F0((__int64)&v72, v74, v24, 8u, v17, v16);
                v23 = (unsigned int)v73;
              }
              *(_QWORD *)&v72[8 * v23] = v25;
              LODWORD(v73) = v73 + 1;
            }
            v19 += 4;
          }
          while ( v20 != v19 );
          v2 = v54;
          v3 = v22;
          v8 = v21;
        }
        sub_D5F1F0(*(_QWORD *)(v2 + 192), v3);
        v37 = sub_B4DE20(v3);
        v55 = *(_QWORD *)(v3 + 72);
        v56 = *(unsigned int ***)(v2 + 192);
        v65 = sub_BD5D20(v58);
        v66 = v38;
        v67 = ".sroa.gep";
        v68 = 773;
        v59 = sub_921130(v56, v55, *(_QWORD *)v69, (_BYTE **)v69 + 1, (unsigned int)v70 - 1LL, (__int64)&v65, v37);
        v57 = *(unsigned int ***)(v2 + 192);
        v65 = sub_BD5D20(v61);
        v66 = v39;
        v67 = ".sroa.gep";
        v68 = 773;
        v40 = sub_921130(v57, v55, *(_QWORD *)v72, (_BYTE **)v72 + 1, (unsigned int)v73 - 1LL, (__int64)&v65, v37);
        v62 = *(__int64 **)(v2 + 192);
        v65 = sub_BD5D20(v59);
        v41 = *(_DWORD *)(v3 + 4);
        v67 = ".cast";
        v68 = 773;
        v66 = v42;
        v60 = sub_291D720(v62, v59, *(_QWORD *)(*(_QWORD *)(v3 - 32LL * (v41 & 0x7FFFFFF)) + 8LL), (__int64)&v65);
        v63 = *(__int64 **)(v2 + 192);
        v65 = sub_BD5D20(v40);
        v43 = *(_DWORD *)(v3 + 4);
        v68 = 773;
        v67 = ".cast";
        v66 = v44;
        v45 = sub_291D720(v63, v40, *(_QWORD *)(*(_QWORD *)(v3 - 32LL * (v43 & 0x7FFFFFF)) + 8LL), (__int64)&v65);
        v46 = *(unsigned int ***)(v2 + 192);
        v64 = (__int64)v45;
        v65 = sub_BD5D20((__int64)v8);
        v68 = 773;
        v66 = v47;
        v67 = ".sroa.sel";
        v48 = sub_B36550(v46, *((_QWORD *)v8 - 12), (__int64)v60, v64, (__int64)&v65, 0);
        sub_2916060(v2, v3, v8);
        sub_25DDDB0(v2 + 80, v3);
        sub_BD84D0(v3, v48);
        sub_B43D60((_QWORD *)v3);
        sub_AE6EC0(v2 + 80, v48);
        sub_2914720(v2, v48, v49, v50, v51, v52);
        if ( v72 != v74 )
          _libc_free((unsigned __int64)v72);
        if ( v69 != v71 )
          _libc_free((unsigned __int64)v69);
        return 1;
      }
    }
    else
    {
      v15 = v3 - v11;
      if ( v3 == v3 - v11 )
      {
        v73 = 0x600000000LL;
        v72 = v74;
        v16 = 32LL * (v5 & 0x7FFFFFF);
        goto LABEL_56;
      }
      v14 = v3;
    }
    v28 = (unsigned __int8 **)v15;
    v29 = (unsigned __int8 **)v14;
    v30 = v8;
    v31 = v3;
    do
    {
      v34 = *v28;
      if ( v30 == *v28
        || v34 == *(unsigned __int8 **)(v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF))
        && v30 == sub_BD3990(*v28, (__int64)a2) )
      {
        v35 = (unsigned int)v70;
        v36 = (unsigned int)v70 + 1LL;
        if ( v36 > HIDWORD(v70) )
        {
          a2 = v71;
          sub_C8D5F0((__int64)&v69, v71, v36, 8u, v15, v11);
          v35 = (unsigned int)v70;
        }
        *(_QWORD *)&v69[8 * v35] = v58;
        LODWORD(v70) = v70 + 1;
      }
      else
      {
        v32 = (unsigned int)v70;
        v33 = (unsigned int)v70 + 1LL;
        if ( v33 > HIDWORD(v70) )
        {
          a2 = v71;
          sub_C8D5F0((__int64)&v69, v71, v33, 8u, v15, v11);
          v32 = (unsigned int)v70;
        }
        *(_QWORD *)&v69[8 * v32] = v34;
        LODWORD(v70) = v70 + 1;
      }
      v28 += 4;
    }
    while ( v29 != v28 );
    v3 = v31;
    v2 = a1;
    v8 = v30;
    v53 = *(_DWORD *)(v3 + 4);
    v72 = v74;
    v73 = 0x600000000LL;
    v16 = 32LL * (v53 & 0x7FFFFFF);
    if ( (*(_BYTE *)(v3 + 7) & 0x40) != 0 )
    {
      v13 = *(_QWORD *)(v3 - 8);
      goto LABEL_16;
    }
LABEL_56:
    v18 = v3;
    v17 = v3 - v16;
    goto LABEL_17;
  }
  do
  {
LABEL_4:
    while ( 1 )
    {
      a2 = *v7;
      v9 = **v7;
      if ( v9 > 0x1Cu )
        break;
      if ( v9 != 17 )
        return 0;
      v7 += 4;
      if ( (unsigned __int8 **)v3 == v7 )
        goto LABEL_12;
    }
    if ( v9 != 86 || v8 || **((_BYTE **)a2 - 8) != 17 || **((_BYTE **)a2 - 4) != 17 )
      return 0;
    v7 += 4;
    v8 = a2;
  }
  while ( (unsigned __int8 **)v3 != v7 );
LABEL_12:
  result = 0;
  if ( v8 )
    goto LABEL_13;
  return result;
}
