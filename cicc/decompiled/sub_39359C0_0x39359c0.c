// Function: sub_39359C0
// Address: 0x39359c0
//
__int64 __fastcall sub_39359C0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r15
  __int64 v4; // r12
  char *v5; // r10
  __int64 v6; // rbx
  char *v7; // r12
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned int v11; // eax
  unsigned __int64 v12; // rax
  __int64 result; // rax
  __int64 *v14; // r8
  __int64 *i; // rdx
  int v16; // ecx
  int v17; // edi
  __int64 v18; // r10
  unsigned int v19; // esi
  __int64 *v20; // rcx
  __int64 v21; // r11
  __int64 v22; // rcx
  _BYTE *v23; // rax
  _BYTE *v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rax
  unsigned int v28; // esi
  __int64 v29; // rdx
  __int64 v30; // r9
  unsigned int v31; // r8d
  unsigned int v32; // r10d
  __int64 *v33; // rax
  __int64 v34; // rdi
  int v35; // r11d
  __int64 *v36; // rcx
  unsigned int v37; // r10d
  __int64 *v38; // rax
  __int64 v39; // rdi
  int v40; // edi
  unsigned __int64 v41; // rax
  _BYTE *v42; // rax
  int v43; // r11d
  int v44; // r11d
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // r8d
  __int64 *v49; // r12
  unsigned __int64 v50; // rax
  __int64 v51; // r11
  unsigned int v52; // r12d
  int v53; // r11d
  __int64 *v54; // r12
  __int64 *v55; // r10
  unsigned int v56; // r13d
  int v57; // eax
  __int64 *v58; // rax
  __int64 v59; // rdx
  _QWORD *v60; // rax
  int v61; // ecx
  int v62; // ebx
  int v63; // r10d
  int v64; // r10d
  __int64 v65; // rdx
  __int64 *v66; // r11
  __int64 v67; // r12
  int v68; // r8d
  __int64 v69; // rax
  int v70; // eax
  int v71; // edx
  __int64 v72; // rdi
  unsigned int v73; // esi
  __int64 v74; // rcx
  int v75; // r9d
  __int64 *v76; // r8
  int v77; // eax
  int v78; // edx
  __int64 v79; // rdi
  unsigned int v80; // esi
  int v81; // r9d
  __int64 v82; // rcx
  __int64 v84; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v85; // [rsp+38h] [rbp-88h]
  char *v86; // [rsp+38h] [rbp-88h]
  char *v87; // [rsp+38h] [rbp-88h]
  int v88; // [rsp+38h] [rbp-88h]
  char *v89; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v90; // [rsp+48h] [rbp-78h]
  char *v91; // [rsp+50h] [rbp-70h] BYREF
  __int64 v92; // [rsp+58h] [rbp-68h]
  _QWORD v93[2]; // [rsp+60h] [rbp-60h] BYREF
  const char *v94; // [rsp+70h] [rbp-50h] BYREF
  char **v95; // [rsp+78h] [rbp-48h]
  __int16 v96; // [rsp+80h] [rbp-40h]

  v84 = a2[261];
  if ( v84 != a2[260] )
  {
    v3 = a2[260];
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 + 8);
      v5 = *(char **)v3;
      v6 = *(_QWORD *)(v3 + 16);
      if ( !v4 )
      {
LABEL_4:
        v90 = v4;
        v7 = &v5[v4];
        v89 = v5;
        v91 = v7;
        v92 = 0;
        v85 = 0;
        goto LABEL_5;
      }
      if ( v4 >= 0 )
        break;
      v87 = *(char **)v3;
      v42 = memchr(*(const void **)v3, 64, 0x7FFFFFFFFFFFFFFFuLL);
      v5 = v87;
      v24 = v42;
      v25 = -1;
      if ( v24 )
        goto LABEL_25;
LABEL_26:
      if ( v25 > v4 )
        goto LABEL_4;
      v26 = v4;
      v89 = v5;
      v7 = &v5[v25];
      v90 = v25;
      v85 = v26 - v25;
      if ( v26 - v25 == -1 )
      {
        v91 = &v5[v25];
        v92 = -1;
LABEL_29:
        if ( *(_WORD *)v7 == 16448 && v7[2] == 64 )
        {
          if ( (*(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL) != 0
            || (*(_BYTE *)(v6 + 9) & 0xC) == 8
            && (*(_BYTE *)(v6 + 8) |= 4u,
                v41 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v6 + 24)),
                *(_QWORD *)v6 = v41 | *(_QWORD *)v6 & 7LL,
                v41) )
          {
            v27 = 1;
          }
          else
          {
            v27 = 2;
          }
          v91 = &v7[v27];
          v92 = v85 - v27;
        }
        goto LABEL_5;
      }
      v91 = &v5[v25];
      v92 = v26 - v25;
      if ( v85 > 2 )
        goto LABEL_29;
LABEL_5:
      v8 = *a2;
      v96 = 1285;
      v94 = (const char *)&v89;
      v95 = &v91;
      v9 = sub_38BF510(v8, (__int64)&v94);
      sub_390D5F0((__int64)a2, v9, 0);
      v10 = sub_38CF310(v6, 0, *a2, 0);
      sub_38E2470(v9, v10);
      *(_BYTE *)(v9 + 8) = *(_BYTE *)(v6 + 8) & 0x10 | *(_BYTE *)(v9 + 8) & 0xEF;
      v11 = sub_38E27C0(v6);
      sub_38E2920(v9, v11);
      if ( (*(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        goto LABEL_35;
      if ( (*(_BYTE *)(v6 + 9) & 0xC) != 8 )
        goto LABEL_7;
      *(_BYTE *)(v6 + 8) |= 4u;
      v12 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v6 + 24));
      *(_QWORD *)v6 = v12 | *(_QWORD *)v6 & 7LL;
      if ( v12 )
      {
LABEL_35:
        if ( v85 <= 2 || *(_WORD *)v7 != 16448 || v7[2] != 64 )
          goto LABEL_15;
      }
      if ( (*(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        if ( (*(_BYTE *)(v6 + 9) & 0xC) != 8
          || (*(_BYTE *)(v6 + 8) |= 4u,
              v50 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v6 + 24)),
              *(_QWORD *)v6 = v50 | *(_QWORD *)v6 & 7LL,
              !v50) )
        {
LABEL_7:
          if ( v85 > 1 && *(_WORD *)v7 == 16448 && (v85 == 2 || v7[2] != 64) )
            sub_16BD130("A @@ version cannot be undefined", 1u);
        }
      }
      v28 = *(_DWORD *)(a1 + 72);
      v29 = *(_QWORD *)(a1 + 56);
      v30 = a1 + 48;
      if ( !v28 )
      {
        ++*(_QWORD *)(a1 + 48);
        goto LABEL_54;
      }
      v31 = v28 - 1;
      v32 = (v28 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v33 = (__int64 *)(v29 + 16LL * v32);
      v34 = *v33;
      if ( v6 != *v33 )
      {
        v88 = 1;
        v51 = *v33;
        v52 = (v28 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        do
        {
          if ( v51 == -8 )
            goto LABEL_40;
          v52 = v31 & (v88 + v52);
          ++v88;
          v51 = *(_QWORD *)(v29 + 16LL * v52);
        }
        while ( v6 != v51 );
        v33 = (__int64 *)(v29 + 16LL * (v31 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4))));
        v53 = 1;
        v54 = 0;
        while ( v34 != -8 )
        {
          if ( v34 == -16 && !v54 )
            v54 = v33;
          v32 = v31 & (v53 + v32);
          v33 = (__int64 *)(v29 + 16LL * v32);
          v34 = *v33;
          if ( v6 == *v33 )
            goto LABEL_39;
          ++v53;
        }
        v55 = v54;
        v56 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
        if ( !v54 )
          v55 = v33;
        ++*(_QWORD *)(a1 + 48);
        v57 = *(_DWORD *)(a1 + 64) + 1;
        if ( 4 * v57 >= 3 * v28 )
        {
          sub_3935800(v30, 2 * v28);
          v70 = *(_DWORD *)(a1 + 72);
          if ( v70 )
          {
            v71 = v70 - 1;
            v72 = *(_QWORD *)(a1 + 56);
            v73 = v71 & v56;
            v57 = *(_DWORD *)(a1 + 64) + 1;
            v55 = (__int64 *)(v72 + 16LL * (v71 & v56));
            v74 = *v55;
            if ( v6 == *v55 )
              goto LABEL_73;
            v75 = 1;
            v76 = 0;
            while ( v74 != -8 )
            {
              if ( !v76 && v74 == -16 )
                v76 = v55;
              v73 = v71 & (v75 + v73);
              v55 = (__int64 *)(v72 + 16LL * v73);
              v74 = *v55;
              if ( v6 == *v55 )
                goto LABEL_73;
              ++v75;
            }
LABEL_93:
            if ( v76 )
              v55 = v76;
            goto LABEL_73;
          }
        }
        else
        {
          if ( v28 - (v57 + *(_DWORD *)(a1 + 68)) > v28 >> 3 )
          {
LABEL_73:
            *(_DWORD *)(a1 + 64) = v57;
            if ( *v55 != -8 )
              --*(_DWORD *)(a1 + 68);
            *v55 = v6;
            v55[1] = 0;
LABEL_76:
            if ( (*(_BYTE *)v6 & 4) != 0 )
            {
              v58 = *(__int64 **)(v6 - 8);
              v59 = *v58;
              v60 = v58 + 2;
            }
            else
            {
              v59 = 0;
              v60 = 0;
            }
            v93[0] = v60;
            v94 = "Multiple symbol versions defined for ";
            v95 = (char **)v93;
            v93[1] = v59;
            v96 = 1283;
            sub_16BCFB0((__int64)&v94, 1u);
          }
          sub_3935800(v30, v28);
          v77 = *(_DWORD *)(a1 + 72);
          if ( v77 )
          {
            v78 = v77 - 1;
            v79 = *(_QWORD *)(a1 + 56);
            v80 = v78 & v56;
            v81 = 1;
            v76 = 0;
            v57 = *(_DWORD *)(a1 + 64) + 1;
            v55 = (__int64 *)(v79 + 16LL * (v78 & v56));
            v82 = *v55;
            if ( v6 == *v55 )
              goto LABEL_73;
            while ( v82 != -8 )
            {
              if ( v82 == -16 && !v76 )
                v76 = v55;
              v80 = v78 & (v81 + v80);
              v55 = (__int64 *)(v79 + 16LL * v80);
              v82 = *v55;
              if ( v6 == *v55 )
                goto LABEL_73;
              ++v81;
            }
            goto LABEL_93;
          }
        }
LABEL_132:
        ++*(_DWORD *)(a1 + 64);
        BUG();
      }
LABEL_39:
      if ( v9 != v33[1] )
        goto LABEL_76;
LABEL_40:
      v35 = 1;
      v36 = 0;
      v37 = v31 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v38 = (__int64 *)(v29 + 16LL * v37);
      v39 = *v38;
      if ( v6 != *v38 )
      {
        while ( v39 != -8 )
        {
          if ( v36 || v39 != -16 )
            v38 = v36;
          v37 = v31 & (v35 + v37);
          v39 = *(_QWORD *)(v29 + 16LL * v37);
          if ( v6 == v39 )
            goto LABEL_15;
          ++v35;
          v36 = v38;
          v38 = (__int64 *)(v29 + 16LL * v37);
        }
        if ( !v36 )
          v36 = v38;
        ++*(_QWORD *)(a1 + 48);
        v40 = *(_DWORD *)(a1 + 64) + 1;
        if ( 4 * v40 < 3 * v28 )
        {
          if ( v28 - *(_DWORD *)(a1 + 68) - v40 <= v28 >> 3 )
          {
            sub_3935800(v30, v28);
            v63 = *(_DWORD *)(a1 + 72);
            if ( !v63 )
              goto LABEL_132;
            v64 = v63 - 1;
            v65 = *(_QWORD *)(a1 + 56);
            v66 = 0;
            LODWORD(v67) = v64 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
            v68 = 1;
            v40 = *(_DWORD *)(a1 + 64) + 1;
            v36 = (__int64 *)(v65 + 16LL * (unsigned int)v67);
            v69 = *v36;
            if ( v6 != *v36 )
            {
              while ( v69 != -8 )
              {
                if ( v69 == -16 && !v66 )
                  v66 = v36;
                v67 = v64 & (unsigned int)(v67 + v68);
                v36 = (__int64 *)(v65 + 16 * v67);
                v69 = *v36;
                if ( v6 == *v36 )
                  goto LABEL_46;
                ++v68;
              }
              if ( v66 )
                v36 = v66;
            }
          }
          goto LABEL_46;
        }
LABEL_54:
        sub_3935800(v30, 2 * v28);
        v43 = *(_DWORD *)(a1 + 72);
        if ( !v43 )
          goto LABEL_132;
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a1 + 56);
        LODWORD(v46) = v44 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v40 = *(_DWORD *)(a1 + 64) + 1;
        v36 = (__int64 *)(v45 + 16LL * (unsigned int)v46);
        v47 = *v36;
        if ( v6 != *v36 )
        {
          v48 = 1;
          v49 = 0;
          while ( v47 != -8 )
          {
            if ( v47 == -16 && !v49 )
              v49 = v36;
            v46 = v44 & (unsigned int)(v46 + v48);
            v36 = (__int64 *)(v45 + 16 * v46);
            v47 = *v36;
            if ( v6 == *v36 )
              goto LABEL_46;
            ++v48;
          }
          if ( v49 )
            v36 = v49;
        }
LABEL_46:
        *(_DWORD *)(a1 + 64) = v40;
        if ( *v36 != -8 )
          --*(_DWORD *)(a1 + 68);
        *v36 = v6;
        v36[1] = v9;
      }
LABEL_15:
      v3 += 24;
      if ( v84 == v3 )
        goto LABEL_16;
    }
    v86 = *(char **)v3;
    v23 = memchr(*(const void **)v3, 64, *(_QWORD *)(v3 + 8));
    v5 = v86;
    v24 = v23;
    if ( !v23 )
      goto LABEL_4;
LABEL_25:
    v25 = v24 - v5;
    goto LABEL_26;
  }
LABEL_16:
  result = a1;
  v14 = *(__int64 **)(a1 + 96);
  for ( i = *(__int64 **)(a1 + 88); i != v14; *(_BYTE *)(result + 9) |= 2u )
  {
    v16 = *(_DWORD *)(a1 + 72);
    result = *i;
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 56);
      v19 = (v16 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( result == *v20 )
      {
LABEL_19:
        v22 = v20[1];
        if ( v22 )
        {
          *i = v22;
          result = v22;
        }
      }
      else
      {
        v61 = 1;
        while ( v21 != -8 )
        {
          v62 = v61 + 1;
          v19 = v17 & (v61 + v19);
          v20 = (__int64 *)(v18 + 16LL * v19);
          v21 = *v20;
          if ( result == *v20 )
            goto LABEL_19;
          v61 = v62;
        }
      }
    }
    ++i;
  }
  return result;
}
