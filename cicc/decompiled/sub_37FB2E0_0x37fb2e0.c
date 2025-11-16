// Function: sub_37FB2E0
// Address: 0x37fb2e0
//
void __fastcall sub_37FB2E0(unsigned __int16 *a1, __int64 a2, unsigned __int16 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r8
  __int64 v8; // r12
  __int64 v9; // rax
  int v10; // r14d
  const void *v11; // rsi
  unsigned __int64 v12; // rcx
  unsigned __int16 v13; // r13
  __int16 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  int v18; // r14d
  unsigned __int16 v19; // ax
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  int v34; // eax
  unsigned __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // eax
  int v39; // eax
  __int64 v40; // rdx
  int v41; // eax
  int v42; // edx
  int v43; // ecx
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rax
  int v49; // eax
  char v50; // al
  __int64 v51; // rdx
  __int64 v52; // rax
  unsigned __int64 v53; // rcx
  __int64 v54; // rax
  unsigned __int64 v55; // rcx
  __int64 v56; // rax
  unsigned __int64 v57; // rcx
  __int64 v58; // rax
  unsigned __int64 v59; // rcx
  __int64 v60; // rax
  __int64 v61; // rax
  int v62; // eax
  unsigned __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rax
  _BOOL8 v66; // rax
  int v67; // eax
  int v68; // eax
  __int64 v69; // rdi
  int v70; // edx
  int v71; // eax
  unsigned __int64 v72; // rcx
  __int64 v73; // rax
  __int64 v74; // rax
  int v75; // eax
  __int64 v76; // rax
  unsigned __int64 v77; // rcx
  __int64 v78; // rax
  _BOOL4 v79; // [rsp+0h] [rbp-70h]
  int v80; // [rsp+0h] [rbp-70h]
  __int64 v81; // [rsp+0h] [rbp-70h]
  __int64 v82; // [rsp+8h] [rbp-68h]
  __int64 v83; // [rsp+8h] [rbp-68h]
  __int64 v84; // [rsp+8h] [rbp-68h]
  __int64 v85; // [rsp+8h] [rbp-68h]
  __int64 v86; // [rsp+8h] [rbp-68h]
  __int64 v87; // [rsp+8h] [rbp-68h]
  __int64 v88; // [rsp+8h] [rbp-68h]
  __int64 v89; // [rsp+8h] [rbp-68h]
  __int64 v90; // [rsp+8h] [rbp-68h]
  __int64 v91; // [rsp+10h] [rbp-60h]
  __int64 v92; // [rsp+18h] [rbp-58h]
  __int64 v93; // [rsp+20h] [rbp-50h]
  __int64 v94; // [rsp+28h] [rbp-48h]
  __int64 v95; // [rsp+30h] [rbp-40h]
  int v96; // [rsp+38h] [rbp-38h]

  if ( a3 > 0x151Du )
  {
    switch ( a3 )
    {
      case 0x1601u:
        v52 = *(unsigned int *)(a4 + 8);
        v53 = *(unsigned int *)(a4 + 12);
        v91 = 1;
        LODWORD(v92) = 1;
        if ( v52 + 1 > v53 )
        {
          sub_C8D5F0(a4, (const void *)(a4 + 16), v52 + 1, 0xCu, a5, a6);
          v52 = *(unsigned int *)(a4 + 8);
        }
        v22 = 0x400000000LL;
        v54 = *(_QWORD *)a4 + 12 * v52;
        *(_QWORD *)v54 = v91;
        *(_DWORD *)(v54 + 8) = v92;
        v24 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
        *(_DWORD *)(a4 + 8) = v24;
        goto LABEL_65;
      case 0x1602u:
LABEL_53:
        v91 = 0;
LABEL_54:
        LODWORD(v92) = 2;
        goto LABEL_50;
      case 0x1603u:
        if ( !*a1 )
          return;
        LODWORD(v92) = *a1;
        v91 = 0x200000001LL;
        break;
      case 0x1604u:
        if ( !*(_DWORD *)a1 )
          return;
        LODWORD(v92) = *(_DWORD *)a1;
        v91 = 0x400000001LL;
        break;
      case 0x1605u:
        v91 = 1;
        LODWORD(v92) = 1;
        goto LABEL_50;
      case 0x1606u:
        v56 = *(unsigned int *)(a4 + 8);
        v57 = *(unsigned int *)(a4 + 12);
        v91 = 0;
        LODWORD(v92) = 1;
        if ( v56 + 1 > v57 )
        {
          sub_C8D5F0(a4, (const void *)(a4 + 16), v56 + 1, 0xCu, a5, a6);
          v56 = *(unsigned int *)(a4 + 8);
        }
        v22 = 0x400000001LL;
        v58 = *(_QWORD *)a4 + 12 * v56;
        *(_QWORD *)v58 = v91;
        *(_DWORD *)(v58 + 8) = v92;
        v24 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
        *(_DWORD *)(a4 + 8) = v24;
        goto LABEL_65;
      case 0x1607u:
        goto LABEL_49;
      default:
        return;
    }
    goto LABEL_50;
  }
  if ( a3 > 0x1502u )
  {
    switch ( a3 )
    {
      case 0x1503u:
      case 0x151Du:
        goto LABEL_53;
      case 0x1504u:
      case 0x1505u:
      case 0x1519u:
        LODWORD(v92) = 3;
        v91 = 0x400000000LL;
        goto LABEL_50;
      case 0x1506u:
        LODWORD(v92) = 1;
        v91 = 0x400000000LL;
        goto LABEL_50;
      case 0x1507u:
        v91 = 0x400000000LL;
        goto LABEL_54;
      default:
        return;
    }
    return;
  }
  if ( a3 == 4609 )
  {
    if ( !*(_DWORD *)a1 )
      return;
    LODWORD(v92) = *(_DWORD *)a1;
    v91 = 0x400000000LL;
LABEL_50:
    v24 = *(unsigned int *)(a4 + 8);
    v44 = v24 + 1;
    if ( v24 + 1 <= (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
LABEL_51:
      v29 = 3 * v24;
LABEL_52:
      v45 = *(_QWORD *)a4 + 4 * v29;
      *(_QWORD *)v45 = v91;
      *(_DWORD *)(v45 + 8) = v92;
      ++*(_DWORD *)(a4 + 8);
      return;
    }
LABEL_66:
    sub_C8D5F0(a4, (const void *)(a4 + 16), v44, 0xCu, a5, a6);
    v24 = *(unsigned int *)(a4 + 8);
    goto LABEL_51;
  }
  if ( a3 > 0x1201u )
  {
    if ( a3 != 4613 )
    {
      v7 = (__int64)a1;
      if ( a3 == 4614 )
      {
        v8 = a2;
        if ( a2 )
        {
          v9 = *(unsigned int *)(a4 + 8);
          v10 = 0;
          v11 = (const void *)(a4 + 16);
          do
          {
            v12 = *(unsigned int *)(a4 + 12);
            v13 = *(_WORD *)v7;
            LODWORD(v91) = 0;
            HIDWORD(v91) = v10 + 4;
            LODWORD(v92) = 1;
            if ( v9 + 1 > v12 )
            {
              v81 = v7;
              sub_C8D5F0(a4, v11, v9 + 1, 0xCu, v7, a6);
              v9 = *(unsigned int *)(a4 + 8);
              v7 = v81;
            }
            v14 = (v13 >> 2) & 5;
            v15 = *(_QWORD *)a4 + 12 * v9;
            *(_QWORD *)v15 = v91;
            *(_DWORD *)(v15 + 8) = v92;
            v9 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
            *(_DWORD *)(a4 + 8) = v9;
            v16 = 4LL * ((_BYTE)v14 == 4) + 8;
            v10 += 4 * ((_BYTE)v14 == 4) + 8;
            v7 += v16;
            v8 -= v16;
          }
          while ( v8 );
        }
        return;
      }
      if ( a3 != 4611 )
        return;
      v17 = a2;
      v18 = 0;
      if ( !a2 )
        return;
      while ( 1 )
      {
        v19 = *(_WORD *)v7;
        if ( *(_WORD *)v7 > 0x1511u )
          return;
        if ( v19 > 0x1501u )
        {
          switch ( v19 )
          {
            case 0x1502u:
              v68 = *(unsigned __int16 *)(v7 + 4);
              if ( (v68 & 0x8000u) == 0 )
              {
                v69 = 6;
                v70 = 6;
              }
              else
              {
                v91 = 0x200000001LL;
                v92 = 0x400000002LL;
                v93 = 0x400000004LL;
                v94 = 0xA00000008LL;
                v95 = 0x800000010LL;
                v96 = 8;
                v69 = (unsigned int)(*((_DWORD *)&v91 + v68 - 0x8000) + 6);
                v70 = *((_DWORD *)&v91 + v68 - 0x8000) + 6;
              }
              goto LABEL_87;
            case 0x150Du:
              v72 = *(unsigned int *)(a4 + 12);
              LODWORD(v91) = 0;
              HIDWORD(v91) = v18 + 4;
              v73 = *(unsigned int *)(a4 + 8);
              LODWORD(v92) = 1;
              if ( v73 + 1 > v72 )
              {
                v90 = v7;
                sub_C8D5F0(a4, (const void *)(a4 + 16), v73 + 1, 0xCu, v7, a6);
                v73 = *(unsigned int *)(a4 + 8);
                v7 = v90;
              }
              v74 = *(_QWORD *)a4 + 12 * v73;
              *(_QWORD *)v74 = v91;
              *(_DWORD *)(v74 + 8) = v92;
              ++*(_DWORD *)(a4 + 8);
              v75 = *(unsigned __int16 *)(v7 + 8);
              if ( (v75 & 0x8000u) == 0 )
              {
                v69 = 10;
                v70 = 10;
              }
              else
              {
                v91 = 0x200000001LL;
                v92 = 0x400000002LL;
                v93 = 0x400000004LL;
                v94 = 0xA00000008LL;
                v95 = 0x800000010LL;
                v96 = 8;
                v69 = (unsigned int)(*((_DWORD *)&v91 + v75 - 0x8000) + 10);
                v70 = *((_DWORD *)&v91 + v75 - 0x8000) + 10;
              }
LABEL_87:
              v80 = v70;
              v84 = v7;
              v71 = strlen((const char *)(v7 + v69));
              v7 = v84;
              v33 = (unsigned int)(v80 + v71 + 1);
              v34 = v80 + v71 + 1;
              goto LABEL_58;
            case 0x150Eu:
            case 0x150Fu:
            case 0x1510u:
              v46 = *(unsigned int *)(a4 + 12);
              LODWORD(v91) = 0;
              HIDWORD(v91) = v18 + 4;
              v47 = *(unsigned int *)(a4 + 8);
              LODWORD(v92) = 1;
              if ( v47 + 1 > v46 )
              {
                v86 = v7;
                sub_C8D5F0(a4, (const void *)(a4 + 16), v47 + 1, 0xCu, v7, a6);
                v47 = *(unsigned int *)(a4 + 8);
                v7 = v86;
              }
              v48 = *(_QWORD *)a4 + 12 * v47;
              *(_QWORD *)v48 = v91;
              *(_DWORD *)(v48 + 8) = v92;
              ++*(_DWORD *)(a4 + 8);
              v82 = v7;
              v49 = strlen((const char *)(v7 + 8));
              v7 = v82;
              v33 = (unsigned int)(v49 + 9);
              v34 = v49 + 9;
              goto LABEL_58;
            case 0x1511u:
              v63 = *(unsigned int *)(a4 + 12);
              LODWORD(v91) = 0;
              HIDWORD(v91) = v18 + 4;
              v64 = *(unsigned int *)(a4 + 8);
              LODWORD(v92) = 1;
              if ( v64 + 1 > v63 )
              {
                v89 = v7;
                sub_C8D5F0(a4, (const void *)(a4 + 16), v64 + 1, 0xCu, v7, a6);
                v64 = *(unsigned int *)(a4 + 8);
                v7 = v89;
              }
              v83 = v7;
              v65 = *(_QWORD *)a4 + 12 * v64;
              *(_QWORD *)v65 = v91;
              *(_DWORD *)(v65 + 8) = v92;
              ++*(_DWORD *)(a4 + 8);
              v66 = ((*(_WORD *)(v7 + 2) >> 2) & 5) == 4;
              v79 = 4 * v66 + 9;
              v67 = strlen((const char *)(v7 + 4 * v66 + 8));
              v7 = v83;
              v33 = (unsigned int)(v67 + v79);
              v34 = v67 + v79;
              goto LABEL_58;
            default:
              return;
          }
        }
        if ( v19 != 5124 )
        {
          if ( v19 <= 0x1404u )
          {
            if ( v19 == 5120 )
            {
              v59 = *(unsigned int *)(a4 + 12);
              LODWORD(v91) = 0;
              HIDWORD(v91) = v18 + 4;
              v60 = *(unsigned int *)(a4 + 8);
              LODWORD(v92) = 1;
              if ( v60 + 1 > v59 )
              {
                v88 = v7;
                sub_C8D5F0(a4, (const void *)(a4 + 16), v60 + 1, 0xCu, v7, a6);
                v60 = *(unsigned int *)(a4 + 8);
                v7 = v88;
              }
              v61 = *(_QWORD *)a4 + 12 * v60;
              *(_QWORD *)v61 = v91;
              *(_DWORD *)(v61 + 8) = v92;
              ++*(_DWORD *)(a4 + 8);
              v62 = *(unsigned __int16 *)(v7 + 8);
              if ( (v62 & 0x8000u) == 0 )
              {
                v33 = 10;
                v34 = 10;
              }
              else
              {
                v91 = 0x200000001LL;
                v92 = 0x400000002LL;
                v93 = 0x400000004LL;
                v94 = 0xA00000008LL;
                v95 = 0x800000010LL;
                v96 = 8;
                v33 = (unsigned int)(*((_DWORD *)&v91 + v62 - 0x8000) + 10);
                v34 = *((_DWORD *)&v91 + v62 - 0x8000) + 10;
              }
            }
            else
            {
              if ( (unsigned __int16)(v19 - 5121) > 1u )
                return;
              v35 = *(unsigned int *)(a4 + 12);
              LODWORD(v91) = 0;
              HIDWORD(v91) = v18 + 4;
              v36 = *(unsigned int *)(a4 + 8);
              LODWORD(v92) = 2;
              if ( v36 + 1 > v35 )
              {
                v87 = v7;
                sub_C8D5F0(a4, (const void *)(a4 + 16), v36 + 1, 0xCu, v7, a6);
                v36 = *(unsigned int *)(a4 + 8);
                v7 = v87;
              }
              v37 = *(_QWORD *)a4 + 12 * v36;
              *(_QWORD *)v37 = v91;
              *(_DWORD *)(v37 + 8) = v92;
              ++*(_DWORD *)(a4 + 8);
              v38 = *(unsigned __int16 *)(v7 + 12);
              if ( (v38 & 0x8000u) == 0 )
              {
                v40 = 14;
                v41 = 14;
              }
              else
              {
                v39 = v38 - 0x8000;
                v91 = 0x200000001LL;
                v92 = 0x400000002LL;
                v93 = 0x400000004LL;
                v94 = 0xA00000008LL;
                v95 = 0x800000010LL;
                v96 = 8;
                v40 = (unsigned int)(*((_DWORD *)&v91 + v39) + 14);
                v41 = *((_DWORD *)&v91 + v39) + 14;
              }
              v42 = *(unsigned __int16 *)(v7 + v40);
              v43 = 2;
              if ( (v42 & 0x8000u) != 0 )
              {
                v91 = 0x200000001LL;
                v92 = 0x400000002LL;
                v93 = 0x400000004LL;
                v94 = 0xA00000008LL;
                v95 = 0x800000010LL;
                v96 = 8;
                v43 = *((_DWORD *)&v91 + v42 - 0x8000) + 2;
              }
              v33 = (unsigned int)(v41 + v43);
              v34 = v41 + v43;
            }
            goto LABEL_58;
          }
          if ( v19 != 5129 )
            return;
        }
        v30 = *(unsigned int *)(a4 + 12);
        LODWORD(v91) = 0;
        HIDWORD(v91) = v18 + 4;
        v31 = *(unsigned int *)(a4 + 8);
        LODWORD(v92) = 1;
        if ( v31 + 1 > v30 )
        {
          v85 = v7;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v31 + 1, 0xCu, v7, a6);
          v31 = *(unsigned int *)(a4 + 8);
          v7 = v85;
        }
        v32 = *(_QWORD *)a4 + 12 * v31;
        *(_QWORD *)v32 = v91;
        *(_DWORD *)(v32 + 8) = v92;
        v33 = 8;
        v34 = 8;
        ++*(_DWORD *)(a4 + 8);
LABEL_58:
        v17 -= v33;
        if ( v17 )
        {
          v7 += v33;
          v18 += v34;
          v50 = *(_BYTE *)v7;
          if ( *(_BYTE *)v7 <= 0xEFu )
            continue;
          v51 = v50 & 0xF;
          v18 += v50 & 0xF;
          v7 += v51;
          v17 -= v51;
          if ( v17 )
            continue;
        }
        return;
      }
    }
    goto LABEL_49;
  }
  if ( a3 == 4104 )
  {
    v76 = *(unsigned int *)(a4 + 8);
    v77 = *(unsigned int *)(a4 + 12);
    v91 = 0;
    LODWORD(v92) = 1;
    if ( v76 + 1 > v77 )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v76 + 1, 0xCu, a5, a6);
      v76 = *(unsigned int *)(a4 + 8);
    }
    v22 = 0x800000000LL;
    v78 = *(_QWORD *)a4 + 12 * v76;
    *(_QWORD *)v78 = v91;
    *(_DWORD *)(v78 + 8) = v92;
    v24 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
    *(_DWORD *)(a4 + 8) = v24;
    goto LABEL_65;
  }
  if ( a3 > 0x1008u )
  {
    if ( a3 != 4105 )
      return;
    v20 = *(unsigned int *)(a4 + 8);
    v21 = *(unsigned int *)(a4 + 12);
    v91 = 0;
    LODWORD(v92) = 3;
    if ( v20 + 1 > v21 )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v20 + 1, 0xCu, a5, a6);
      v20 = *(unsigned int *)(a4 + 8);
    }
    v22 = 0x1000000000LL;
    v23 = *(_QWORD *)a4 + 12 * v20;
    *(_QWORD *)v23 = v91;
    *(_DWORD *)(v23 + 8) = v92;
    v24 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
    *(_DWORD *)(a4 + 8) = v24;
LABEL_65:
    v55 = *(unsigned int *)(a4 + 12);
    v44 = v24 + 1;
    v91 = v22;
    LODWORD(v92) = 1;
    if ( v24 + 1 <= v55 )
      goto LABEL_51;
    goto LABEL_66;
  }
  if ( a3 == 4097 )
  {
LABEL_49:
    v91 = 0;
    LODWORD(v92) = 1;
    goto LABEL_50;
  }
  if ( a3 == 4098 )
  {
    v25 = *(unsigned int *)(a4 + 8);
    v26 = *(unsigned int *)(a4 + 12);
    v91 = 0;
    LODWORD(v92) = 1;
    if ( v25 + 1 > v26 )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v25 + 1, 0xCu, a5, a6);
      v25 = *(unsigned int *)(a4 + 8);
    }
    v27 = *(_QWORD *)a4 + 12 * v25;
    *(_QWORD *)v27 = v91;
    *(_DWORD *)(v27 + 8) = v92;
    v28 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
    *(_DWORD *)(a4 + 8) = v28;
    if ( (unsigned __int8)(((unsigned __int8)*((_DWORD *)a1 + 1) >> 5) - 2) <= 1u )
    {
      LODWORD(v92) = 1;
      v91 = 0x800000000LL;
      if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v28 + 1, 0xCu, v28 + 1, a6);
        v28 = *(unsigned int *)(a4 + 8);
      }
      v29 = 3 * v28;
      goto LABEL_52;
    }
  }
}
