// Function: sub_1A7AAD0
// Address: 0x1a7aad0
//
void __fastcall sub_1A7AAD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // r14
  __int64 *v8; // rax
  __int64 *v9; // rdx
  char v10; // dl
  __int64 v11; // rax
  __int64 *v12; // rsi
  __int64 *v13; // rcx
  _BYTE *v14; // rdx
  unsigned __int64 v15; // r12
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // r13
  unsigned __int8 v19; // cl
  __int64 *v20; // rax
  __int64 i; // r15
  char v22; // dl
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 *v25; // rsi
  __int64 *v26; // rcx
  unsigned __int64 v27; // r14
  __int64 v28; // r8
  unsigned __int64 v29; // rdx
  __int64 v30; // r15
  unsigned __int64 *v31; // rax
  _QWORD *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r12
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r15
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  char v44; // al
  unsigned __int64 *v45; // rsi
  unsigned int v46; // edi
  unsigned __int64 *v47; // rcx
  __int64 *v48; // rsi
  unsigned int v49; // edi
  __int64 *v50; // rcx
  __int64 *v51; // rax
  char v52; // al
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r12
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r15
  __int64 v61; // rax
  unsigned __int64 *v62; // rax
  __int64 *v63; // rsi
  unsigned int v64; // edi
  __int64 *v65; // rcx
  _BYTE *v66; // r14
  int v67; // r15d
  unsigned __int64 *v68; // rdi
  unsigned int v69; // ecx
  unsigned __int64 *v70; // rdx
  _BYTE *v71; // [rsp+18h] [rbp-288h]
  char v72; // [rsp+18h] [rbp-288h]
  char v73; // [rsp+18h] [rbp-288h]
  char v74; // [rsp+18h] [rbp-288h]
  __int64 v75; // [rsp+28h] [rbp-278h] BYREF
  _BYTE *v76; // [rsp+30h] [rbp-270h] BYREF
  __int64 v77; // [rsp+38h] [rbp-268h]
  _BYTE v78[256]; // [rsp+40h] [rbp-260h] BYREF
  __int64 v79; // [rsp+140h] [rbp-160h] BYREF
  __int64 *v80; // [rsp+148h] [rbp-158h]
  __int64 *v81; // [rsp+150h] [rbp-150h]
  __int64 v82; // [rsp+158h] [rbp-148h]
  int v83; // [rsp+160h] [rbp-140h]
  _BYTE v84[312]; // [rsp+168h] [rbp-138h] BYREF

  v7 = *(_QWORD *)(a2 + 8);
  v76 = v78;
  v77 = 0x2000000000LL;
  v8 = (__int64 *)v84;
  v79 = 0;
  v9 = (__int64 *)v84;
  v80 = (__int64 *)v84;
  v81 = (__int64 *)v84;
  v82 = 32;
  v83 = 0;
  if ( !v7 )
  {
    v66 = v78;
    v67 = 0;
    goto LABEL_19;
  }
  while ( 1 )
  {
    if ( v9 != v8 )
      goto LABEL_3;
    v12 = &v8[HIDWORD(v82)];
    if ( v12 != v8 )
    {
      v13 = 0;
      while ( *v8 != v7 )
      {
        if ( *v8 == -2 )
          v13 = v8;
        if ( v12 == ++v8 )
        {
          if ( !v13 )
            goto LABEL_95;
          *v13 = v7;
          v11 = (unsigned int)v77;
          --v83;
          ++v79;
          if ( (unsigned int)v77 < HIDWORD(v77) )
            goto LABEL_5;
          goto LABEL_17;
        }
      }
      goto LABEL_6;
    }
LABEL_95:
    if ( HIDWORD(v82) < (unsigned int)v82 )
    {
      ++HIDWORD(v82);
      *v12 = v7;
      ++v79;
    }
    else
    {
LABEL_3:
      sub_16CCBA0((__int64)&v79, v7);
      if ( !v10 )
        goto LABEL_6;
    }
    v11 = (unsigned int)v77;
    if ( (unsigned int)v77 >= HIDWORD(v77) )
    {
LABEL_17:
      sub_16CD150((__int64)&v76, v78, 0, 8, a5, a6);
      v11 = (unsigned int)v77;
    }
LABEL_5:
    *(_QWORD *)&v76[8 * v11] = v7;
    LODWORD(v77) = v77 + 1;
LABEL_6:
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      break;
    v9 = v81;
    v8 = v80;
  }
  v66 = v76;
  v67 = v77;
LABEL_19:
  while ( 1 )
  {
    v14 = &v66[8 * v67];
    if ( !v67 )
      break;
    while ( 2 )
    {
      v15 = *((_QWORD *)v14 - 1);
      --v67;
      v71 = v14;
      LODWORD(v77) = v67;
      v18 = (__int64)sub_1648700(v15);
      v19 = *(_BYTE *)(v18 + 16);
      switch ( v19 )
      {
        case 0x1Du:
        case 0x4Eu:
          if ( v19 <= 0x17u )
          {
            v27 = 0;
            v28 = 0;
          }
          else if ( v19 == 78 )
          {
            v28 = v18 | 4;
            v27 = v18 & 0xFFFFFFFFFFFFFFF8LL;
          }
          else
          {
            v27 = 0;
            v28 = 0;
            if ( v19 == 29 )
            {
              v28 = v18 & 0xFFFFFFFFFFFFFFFBLL;
              v27 = v18 & 0xFFFFFFFFFFFFFFF8LL;
            }
          }
          if ( (*(_BYTE *)(v27 + 23) & 0x40) != 0 )
          {
            v29 = *(_QWORD *)(v27 - 8);
            LOBYTE(v30) = 0;
            if ( v15 < v29 )
              goto LABEL_45;
            v43 = 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
            v30 = (v28 >> 2) & 1;
            if ( ((v28 >> 2) & 1) != 0 )
              goto LABEL_99;
          }
          else
          {
            LOBYTE(v30) = 0;
            v43 = 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
            v29 = v27 - v43;
            if ( v15 < v27 - v43 )
              goto LABEL_45;
            v30 = (v28 >> 2) & 1;
            if ( ((v28 >> 2) & 1) != 0 )
            {
LABEL_99:
              LOBYTE(v30) = 0;
              if ( v15 < v29 + v43 - 24 )
              {
                v74 = v28;
                v52 = sub_139C340(v27, -1431655765 * ((__int64)(v15 - v29) >> 3) + 1, 22);
                LOBYTE(v28) = v74;
                LOBYTE(v30) = v52;
              }
              goto LABEL_45;
            }
          }
          if ( v15 < v29 + v43 - 72 )
          {
            v73 = v28;
            v44 = sub_139C570(v27, -1431655765 * ((__int64)(v15 - v29) >> 3) + 1, 22);
            LOBYTE(v28) = v73;
            LOBYTE(v30) = v44;
            v31 = *(unsigned __int64 **)(a1 + 8);
            if ( *(unsigned __int64 **)(a1 + 16) != v31 )
              goto LABEL_46;
            goto LABEL_79;
          }
LABEL_45:
          v31 = *(unsigned __int64 **)(a1 + 8);
          if ( *(unsigned __int64 **)(a1 + 16) != v31 )
          {
LABEL_46:
            v72 = v28;
            sub_16CCBA0(a1, v27);
            LOBYTE(v28) = v72;
            goto LABEL_47;
          }
LABEL_79:
          v45 = &v31[*(unsigned int *)(a1 + 28)];
          v46 = *(_DWORD *)(a1 + 28);
          if ( v31 == v45 )
            goto LABEL_130;
          v47 = 0;
          do
          {
            if ( v27 == *v31 )
              goto LABEL_47;
            if ( *v31 == -2 )
              v47 = v31;
            ++v31;
          }
          while ( v45 != v31 );
          if ( !v47 )
          {
LABEL_130:
            if ( v46 >= *(_DWORD *)(a1 + 24) )
              goto LABEL_46;
            *(_DWORD *)(a1 + 28) = v46 + 1;
            *v45 = v27;
            ++*(_QWORD *)a1;
          }
          else
          {
            *v47 = v27;
            --*(_DWORD *)(a1 + 32);
            ++*(_QWORD *)a1;
          }
LABEL_47:
          if ( (_BYTE)v30 )
            goto LABEL_72;
          v32 = (_QWORD *)(v27 + 56);
          if ( (v28 & 4) != 0 )
          {
            if ( (unsigned __int8)sub_1560260(v32, -1, 36) )
              goto LABEL_23;
            if ( *(char *)(v27 + 23) >= 0 )
              goto LABEL_152;
            v33 = sub_1648A40(v27);
            v35 = v33 + v34;
            v36 = 0;
            if ( *(char *)(v27 + 23) < 0 )
              v36 = sub_1648A40(v27);
            if ( !(unsigned int)((v35 - v36) >> 4) )
            {
LABEL_152:
              v37 = *(_QWORD *)(v27 - 24);
              if ( !*(_BYTE *)(v37 + 16) )
              {
                v75 = *(_QWORD *)(v37 + 112);
                if ( (unsigned __int8)sub_1560260(&v75, -1, 36) )
                  goto LABEL_23;
              }
            }
            if ( (unsigned __int8)sub_1560260((_QWORD *)(v27 + 56), -1, 37) )
              goto LABEL_23;
            if ( *(char *)(v27 + 23) < 0 )
            {
              v38 = sub_1648A40(v27);
              v40 = v38 + v39;
              v41 = *(char *)(v27 + 23) >= 0 ? 0LL : sub_1648A40(v27);
              if ( v41 != v40 )
              {
                while ( *(_DWORD *)(*(_QWORD *)v41 + 8LL) <= 1u )
                {
                  v41 += 16;
                  if ( v40 == v41 )
                    goto LABEL_63;
                }
                goto LABEL_116;
              }
            }
LABEL_63:
            v42 = *(_QWORD *)(v27 - 24);
            if ( *(_BYTE *)(v42 + 16) )
              goto LABEL_116;
          }
          else
          {
            if ( (unsigned __int8)sub_1560260(v32, -1, 36) )
              goto LABEL_23;
            if ( *(char *)(v27 + 23) >= 0 )
              goto LABEL_153;
            v53 = sub_1648A40(v27);
            v55 = v53 + v54;
            v56 = 0;
            if ( *(char *)(v27 + 23) < 0 )
              v56 = sub_1648A40(v27);
            if ( !(unsigned int)((v55 - v56) >> 4) )
            {
LABEL_153:
              v57 = *(_QWORD *)(v27 - 72);
              if ( !*(_BYTE *)(v57 + 16) )
              {
                v75 = *(_QWORD *)(v57 + 112);
                if ( (unsigned __int8)sub_1560260(&v75, -1, 36) )
                  goto LABEL_23;
              }
            }
            if ( (unsigned __int8)sub_1560260((_QWORD *)(v27 + 56), -1, 37) )
              goto LABEL_23;
            if ( *(char *)(v27 + 23) < 0 )
            {
              v58 = sub_1648A40(v27);
              v60 = v58 + v59;
              v61 = *(char *)(v27 + 23) >= 0 ? 0LL : sub_1648A40(v27);
              if ( v61 != v60 )
              {
                while ( *(_DWORD *)(*(_QWORD *)v61 + 8LL) <= 1u )
                {
                  v61 += 16;
                  if ( v60 == v61 )
                    goto LABEL_133;
                }
                goto LABEL_116;
              }
            }
LABEL_133:
            v42 = *(_QWORD *)(v27 - 72);
            if ( *(_BYTE *)(v42 + 16) )
              goto LABEL_116;
          }
          v75 = *(_QWORD *)(v42 + 112);
          if ( (unsigned __int8)sub_1560260(&v75, -1, 37) )
            goto LABEL_23;
LABEL_116:
          v62 = *(unsigned __int64 **)(a1 + 304);
          if ( *(unsigned __int64 **)(a1 + 312) != v62 )
          {
LABEL_117:
            sub_16CCBA0(a1 + 296, v27);
            goto LABEL_23;
          }
          v68 = &v62[*(unsigned int *)(a1 + 324)];
          v69 = *(_DWORD *)(a1 + 324);
          if ( v62 == v68 )
            goto LABEL_147;
          v70 = 0;
          do
          {
            if ( v27 == *v62 )
              goto LABEL_23;
            if ( *v62 == -2 )
              v70 = v62;
            ++v62;
          }
          while ( v68 != v62 );
          if ( !v70 )
          {
LABEL_147:
            if ( v69 >= *(_DWORD *)(a1 + 320) )
              goto LABEL_117;
            *(_DWORD *)(a1 + 324) = v69 + 1;
            *v68 = v27;
            ++*(_QWORD *)(a1 + 296);
          }
          else
          {
            *v70 = v27;
            --*(_DWORD *)(a1 + 328);
            ++*(_QWORD *)(a1 + 296);
          }
LABEL_23:
          for ( i = *(_QWORD *)(v18 + 8); i; i = *(_QWORD *)(i + 8) )
          {
            v24 = v80;
            if ( v81 == v80 )
            {
              v25 = &v80[HIDWORD(v82)];
              if ( v80 != v25 )
              {
                v26 = 0;
                while ( i != *v24 )
                {
                  if ( *v24 == -2 )
                    v26 = v24;
                  if ( v25 == ++v24 )
                  {
                    if ( !v26 )
                      goto LABEL_73;
                    *v26 = i;
                    v23 = (unsigned int)v77;
                    --v83;
                    ++v79;
                    if ( (unsigned int)v77 < HIDWORD(v77) )
                      goto LABEL_27;
                    goto LABEL_38;
                  }
                }
                continue;
              }
LABEL_73:
              if ( HIDWORD(v82) < (unsigned int)v82 )
              {
                ++HIDWORD(v82);
                *v25 = i;
                ++v79;
LABEL_26:
                v23 = (unsigned int)v77;
                if ( (unsigned int)v77 >= HIDWORD(v77) )
                {
LABEL_38:
                  sub_16CD150((__int64)&v76, v78, 0, 8, v16, v17);
                  v23 = (unsigned int)v77;
                }
LABEL_27:
                *(_QWORD *)&v76[8 * v23] = i;
                LODWORD(v77) = v77 + 1;
                continue;
              }
            }
            sub_16CCBA0((__int64)&v79, i);
            if ( v22 )
              goto LABEL_26;
          }
LABEL_72:
          v66 = v76;
          v67 = v77;
          break;
        case 0x36u:
          v14 = v71 - 8;
          if ( v67 )
            continue;
          goto LABEL_66;
        case 0x37u:
          if ( (unsigned int)sub_1648720(v15) )
            goto LABEL_72;
          v51 = *(__int64 **)(a1 + 304);
          if ( *(__int64 **)(a1 + 312) != v51 )
            goto LABEL_98;
          v63 = &v51[*(unsigned int *)(a1 + 324)];
          v64 = *(_DWORD *)(a1 + 324);
          if ( v51 == v63 )
            goto LABEL_144;
          v65 = 0;
          while ( v18 != *v51 )
          {
            if ( *v51 == -2 )
              v65 = v51;
            if ( v63 == ++v51 )
            {
              if ( v65 )
              {
                *v65 = v18;
                --*(_DWORD *)(a1 + 328);
                ++*(_QWORD *)(a1 + 296);
              }
              else
              {
LABEL_144:
                if ( v64 >= *(_DWORD *)(a1 + 320) )
                {
LABEL_98:
                  sub_16CCBA0(a1 + 296, v18);
                }
                else
                {
                  *(_DWORD *)(a1 + 324) = v64 + 1;
                  *v63 = v18;
                  ++*(_QWORD *)(a1 + 296);
                }
              }
              goto LABEL_72;
            }
          }
          goto LABEL_72;
        case 0x38u:
        case 0x47u:
        case 0x48u:
        case 0x4Du:
        case 0x4Fu:
          goto LABEL_23;
        default:
          v20 = *(__int64 **)(a1 + 304);
          if ( *(__int64 **)(a1 + 312) != v20 )
            goto LABEL_22;
          v48 = &v20[*(unsigned int *)(a1 + 324)];
          v49 = *(_DWORD *)(a1 + 324);
          if ( v20 == v48 )
            goto LABEL_128;
          v50 = 0;
          while ( v18 != *v20 )
          {
            if ( *v20 == -2 )
              v50 = v20;
            if ( v48 == ++v20 )
            {
              if ( v50 )
              {
                *v50 = v18;
                --*(_DWORD *)(a1 + 328);
                ++*(_QWORD *)(a1 + 296);
              }
              else
              {
LABEL_128:
                if ( v49 >= *(_DWORD *)(a1 + 320) )
                {
LABEL_22:
                  sub_16CCBA0(a1 + 296, v18);
                }
                else
                {
                  *(_DWORD *)(a1 + 324) = v49 + 1;
                  *v48 = v18;
                  ++*(_QWORD *)(a1 + 296);
                }
              }
              goto LABEL_23;
            }
          }
          goto LABEL_23;
      }
      break;
    }
  }
LABEL_66:
  if ( v81 != v80 )
  {
    _libc_free((unsigned __int64)v81);
    v66 = v76;
  }
  if ( v66 != v78 )
    _libc_free((unsigned __int64)v66);
}
