// Function: sub_18BFB30
// Address: 0x18bfb30
//
__int64 __fastcall sub_18BFB30(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int v10; // r8d
  unsigned int v11; // edx
  __int64 v12; // rcx
  unsigned __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // r12
  unsigned int v18; // esi
  __int64 v19; // rdi
  unsigned int v20; // ecx
  _QWORD *v21; // rbx
  __int64 v22; // rdx
  _QWORD *v23; // r14
  _QWORD *v24; // rcx
  unsigned __int64 v25; // rsi
  _QWORD *v26; // rax
  char v27; // di
  unsigned __int64 v28; // rdx
  _QWORD *v29; // rdi
  _BOOL4 v30; // r8d
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // r10d
  _QWORD *v34; // r9
  int v35; // edi
  int v36; // edx
  int v38; // ecx
  int v39; // ecx
  __int64 v40; // rdi
  unsigned int v41; // eax
  __int64 v42; // rsi
  int v43; // r10d
  _QWORD *v44; // r8
  int v45; // ecx
  int v46; // ecx
  __int64 v47; // rdi
  int v48; // r10d
  unsigned int v49; // eax
  __int64 v50; // rsi
  __int64 *v51; // r9
  int v52; // r11d
  __int64 *v53; // r10
  int v54; // eax
  char *v55; // rsi
  char *v56; // rsi
  __int64 v57; // rdi
  __int64 *v58; // rax
  __int64 v59; // rbx
  __int64 v60; // r14
  __int64 v61; // r13
  unsigned __int64 v62; // r12
  __int64 v63; // rax
  unsigned int v64; // edx
  __int64 v65; // r8
  __int64 *v66; // r10
  int v67; // esi
  __int64 *v68; // rcx
  _QWORD *v69; // r11
  unsigned int v70; // r12d
  __int64 v71; // rdi
  __int64 *v72; // r9
  int v73; // ecx
  __int64 *v74; // rdx
  __int64 v75; // rax
  int v76; // r13d
  unsigned int v77; // r11d
  __int64 v80; // [rsp+28h] [rbp-A8h]
  _QWORD *v81; // [rsp+38h] [rbp-98h]
  _BOOL4 v82; // [rsp+40h] [rbp-90h]
  _QWORD *v83; // [rsp+40h] [rbp-90h]
  unsigned int v84; // [rsp+40h] [rbp-90h]
  _BYTE *v85; // [rsp+48h] [rbp-88h]
  __int64 v86; // [rsp+48h] [rbp-88h]
  __int64 *v87; // [rsp+50h] [rbp-80h]
  __int64 v88; // [rsp+58h] [rbp-78h]
  __int64 v89; // [rsp+60h] [rbp-70h] BYREF
  __int64 i; // [rsp+68h] [rbp-68h]
  __int64 v91; // [rsp+70h] [rbp-60h]
  unsigned int v92; // [rsp+78h] [rbp-58h]
  _BYTE *v93; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v94; // [rsp+88h] [rbp-48h]
  int v95; // [rsp+8Ch] [rbp-44h]
  _BYTE v96[64]; // [rsp+90h] [rbp-40h] BYREF

  v4 = *a1;
  v91 = 0;
  v92 = 0;
  v5 = v4 + 8;
  v6 = *(_QWORD *)(v4 + 16);
  v7 = 0;
  v89 = 0;
  for ( i = 0; v6 != v5; ++v7 )
    v6 = *(_QWORD *)(v6 + 8);
  sub_18B94A0(a2, v7);
  v95 = 2;
  v93 = v96;
  v80 = *a1 + 8;
  if ( *(_QWORD *)(*a1 + 16) != v80 )
  {
    v88 = *(_QWORD *)(*a1 + 16);
    while ( 1 )
    {
      v94 = 0;
      v8 = v88 - 56;
      if ( !v88 )
        v8 = 0;
      sub_1626560(v8, 19, (__int64)&v93);
      v9 = v94;
      if ( !v94 )
        goto LABEL_5;
      if ( !v92 )
        break;
      v10 = v92 - 1;
      v11 = (v92 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v87 = (__int64 *)(i + 16LL * v11);
      v12 = *v87;
      if ( v8 == *v87 )
        goto LABEL_11;
      v51 = (__int64 *)(i + 16LL * ((v92 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))));
      v52 = 1;
      v53 = 0;
      while ( 1 )
      {
        if ( v12 == -8 )
        {
          if ( !v53 )
            v53 = v51;
          ++v89;
          v54 = v91 + 1;
          v87 = v53;
          if ( 4 * ((int)v91 + 1) < 3 * v92 )
          {
            if ( v92 - HIDWORD(v91) - v54 > v92 >> 3 )
            {
LABEL_80:
              LODWORD(v91) = v54;
              if ( *v87 != -8 )
                --HIDWORD(v91);
              *v87 = v8;
              v87[1] = 0;
LABEL_83:
              v55 = *(char **)(a2 + 8);
              if ( v55 == *(char **)(a2 + 16) )
              {
                sub_18BB570((char **)a2, v55);
                v56 = *(char **)(a2 + 8);
              }
              else
              {
                if ( v55 )
                {
                  memset(v55, 0, 0x70u);
                  v55 = *(char **)(a2 + 8);
                }
                v56 = v55 + 112;
                *(_QWORD *)(a2 + 8) = v56;
              }
              *((_QWORD *)v56 - 14) = v8;
              v57 = sub_1632FA0(*a1);
              v58 = *(__int64 **)(v8 - 24);
              v59 = 1;
              v60 = *v58;
              v61 = *(_QWORD *)(a2 + 8);
              v62 = (unsigned int)sub_15A9FE0(v57, *v58);
              while ( 2 )
              {
                switch ( *(_BYTE *)(v60 + 8) )
                {
                  case 0:
                  case 8:
                  case 0xA:
                  case 0xC:
                  case 0x10:
                    v75 = *(_QWORD *)(v60 + 32);
                    v60 = *(_QWORD *)(v60 + 24);
                    v59 *= v75;
                    continue;
                  case 1:
                    v63 = 16;
                    break;
                  case 2:
                    v63 = 32;
                    break;
                  case 3:
                  case 9:
                    v63 = 64;
                    break;
                  case 4:
                    v63 = 80;
                    break;
                  case 5:
                  case 6:
                    v63 = 128;
                    break;
                  case 7:
                    v63 = 8 * (unsigned int)sub_15A9520(v57, 0);
                    break;
                  case 0xB:
                    v63 = *(_DWORD *)(v60 + 8) >> 8;
                    break;
                  case 0xD:
                    v63 = 8LL * *(_QWORD *)sub_15A9930(v57, v60);
                    break;
                  case 0xE:
                    v86 = *(_QWORD *)(v60 + 32);
                    v63 = 8 * v86 * sub_12BE0A0(v57, *(_QWORD *)(v60 + 24));
                    break;
                  case 0xF:
                    v63 = 8 * (unsigned int)sub_15A9520(v57, *(_DWORD *)(v60 + 8) >> 8);
                    break;
                }
                break;
              }
              *(_QWORD *)(v61 - 104) = v62 * ((v62 + ((unsigned __int64)(v59 * v63 + 7) >> 3) - 1) / v62);
              v87[1] = *(_QWORD *)(a2 + 8) - 112LL;
              v9 = v94;
              goto LABEL_12;
            }
            sub_18BF6B0((__int64)&v89, v92);
            if ( v92 )
            {
              v70 = (v92 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
              v87 = (__int64 *)(i + 16LL * v70);
              v71 = *v87;
              v54 = v91 + 1;
              if ( v8 != *v87 )
              {
                v72 = (__int64 *)(i + 16LL * v70);
                v73 = 1;
                v74 = 0;
                while ( v71 != -8 )
                {
                  if ( v71 == -16 && !v74 )
                    v74 = v72;
                  v70 = (v92 - 1) & (v73 + v70);
                  v72 = (__int64 *)(i + 16LL * v70);
                  v71 = *v72;
                  if ( v8 == *v72 )
                  {
                    v87 = (__int64 *)(i + 16LL * v70);
                    goto LABEL_80;
                  }
                  ++v73;
                }
                if ( !v74 )
                  v74 = v72;
                v87 = v74;
              }
              goto LABEL_80;
            }
LABEL_146:
            LODWORD(v91) = v91 + 1;
            BUG();
          }
LABEL_92:
          sub_18BF6B0((__int64)&v89, 2 * v92);
          if ( v92 )
          {
            v64 = (v92 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v87 = (__int64 *)(i + 16LL * v64);
            v65 = *v87;
            v54 = v91 + 1;
            if ( v8 != *v87 )
            {
              v66 = (__int64 *)(i + 16LL * ((v92 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))));
              v67 = 1;
              v68 = 0;
              while ( v65 != -8 )
              {
                if ( !v68 && v65 == -16 )
                  v68 = v66;
                v64 = (v92 - 1) & (v67 + v64);
                v66 = (__int64 *)(i + 16LL * v64);
                v65 = *v66;
                if ( v8 == *v66 )
                {
                  v87 = (__int64 *)(i + 16LL * v64);
                  goto LABEL_80;
                }
                ++v67;
              }
              if ( !v68 )
                v68 = v66;
              v87 = v68;
            }
            goto LABEL_80;
          }
          goto LABEL_146;
        }
        if ( !v53 && v12 == -16 )
          v53 = v51;
        v76 = v52 + 1;
        v77 = v11 + v52;
        v11 = v10 & v77;
        v51 = (__int64 *)(i + 16LL * (v10 & v77));
        v12 = *v51;
        if ( v8 == *v51 )
          break;
        v52 = v76;
      }
      v87 = (__int64 *)(i + 16LL * (v10 & v77));
LABEL_11:
      if ( !v87[1] )
        goto LABEL_83;
LABEL_12:
      v13 = (unsigned __int64)v93;
      v85 = &v93[8 * v9];
      if ( v93 == v85 )
        goto LABEL_5;
      while ( 2 )
      {
        v14 = *(unsigned int *)(*(_QWORD *)v13 + 8LL);
        v15 = *(_QWORD *)(*(_QWORD *)v13 + 8 * (1 - v14));
        v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v13 - 8 * v14) + 136LL);
        v17 = *(_QWORD **)(v16 + 24);
        if ( *(_DWORD *)(v16 + 32) > 0x40u )
          v17 = (_QWORD *)*v17;
        v18 = *(_DWORD *)(a3 + 24);
        if ( !v18 )
        {
          ++*(_QWORD *)a3;
          goto LABEL_59;
        }
        v19 = *(_QWORD *)(a3 + 8);
        v20 = (v18 - 1) & (((unsigned int)v15 >> 4) ^ ((unsigned int)v15 >> 9));
        v21 = (_QWORD *)(v19 + 56LL * v20);
        v22 = *v21;
        if ( v15 == *v21 )
        {
          v23 = (_QWORD *)v21[3];
          v24 = v21 + 2;
          goto LABEL_18;
        }
        v33 = 1;
        v34 = 0;
        while ( 2 )
        {
          if ( v22 == -4 )
          {
            v35 = *(_DWORD *)(a3 + 16);
            if ( v34 )
              v21 = v34;
            ++*(_QWORD *)a3;
            v36 = v35 + 1;
            if ( 4 * (v35 + 1) < 3 * v18 )
            {
              if ( v18 - *(_DWORD *)(a3 + 20) - v36 > v18 >> 3 )
                goto LABEL_50;
              v84 = ((unsigned int)v15 >> 4) ^ ((unsigned int)v15 >> 9);
              sub_18BF870(a3, v18);
              v45 = *(_DWORD *)(a3 + 24);
              if ( v45 )
              {
                v46 = v45 - 1;
                v47 = *(_QWORD *)(a3 + 8);
                v44 = 0;
                v48 = 1;
                v49 = v46 & v84;
                v21 = (_QWORD *)(v47 + 56LL * (v46 & v84));
                v50 = *v21;
                v36 = *(_DWORD *)(a3 + 16) + 1;
                if ( v15 != *v21 )
                {
                  while ( v50 != -4 )
                  {
                    if ( !v44 && v50 == -8 )
                      v44 = v21;
                    v49 = v46 & (v49 + v48);
                    v21 = (_QWORD *)(v47 + 56LL * v49);
                    v50 = *v21;
                    if ( v15 == *v21 )
                      goto LABEL_50;
                    ++v48;
                  }
                  goto LABEL_63;
                }
                goto LABEL_50;
              }
              goto LABEL_147;
            }
LABEL_59:
            sub_18BF870(a3, 2 * v18);
            v38 = *(_DWORD *)(a3 + 24);
            if ( v38 )
            {
              v39 = v38 - 1;
              v40 = *(_QWORD *)(a3 + 8);
              v41 = v39 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
              v21 = (_QWORD *)(v40 + 56LL * v41);
              v42 = *v21;
              v36 = *(_DWORD *)(a3 + 16) + 1;
              if ( v15 != *v21 )
              {
                v43 = 1;
                v44 = 0;
                while ( v42 != -4 )
                {
                  if ( !v44 && v42 == -8 )
                    v44 = v21;
                  v41 = v39 & (v41 + v43);
                  v21 = (_QWORD *)(v40 + 56LL * v41);
                  v42 = *v21;
                  if ( v15 == *v21 )
                    goto LABEL_50;
                  ++v43;
                }
LABEL_63:
                if ( v44 )
                  v21 = v44;
              }
LABEL_50:
              *(_DWORD *)(a3 + 16) = v36;
              if ( *v21 != -4 )
                --*(_DWORD *)(a3 + 20);
              *v21 = v15;
              v23 = v21 + 2;
              *((_DWORD *)v21 + 4) = 0;
              v24 = v21 + 2;
              v21[3] = 0;
              v21[4] = v21 + 2;
              v21[5] = v21 + 2;
              v21[6] = 0;
              v25 = v87[1];
LABEL_53:
              if ( (_QWORD *)v21[4] != v23 )
              {
LABEL_39:
                v83 = v24;
                v32 = sub_220EF80(v23);
                v24 = v83;
                v28 = *(_QWORD *)(v32 + 32);
                if ( v25 <= v28 )
                {
                  v29 = v23;
                  v23 = (_QWORD *)v32;
                  goto LABEL_28;
                }
LABEL_35:
                if ( !v23 )
                  goto LABEL_29;
                goto LABEL_36;
              }
              v30 = 1;
LABEL_37:
              v81 = v24;
              v82 = v30;
              v31 = sub_22077B0(48);
              *(_QWORD *)(v31 + 40) = v17;
              *(_QWORD *)(v31 + 32) = v25;
              sub_220F040(v82, v31, v23, v81);
              ++v21[6];
              goto LABEL_29;
            }
LABEL_147:
            ++*(_DWORD *)(a3 + 16);
            BUG();
          }
          if ( v22 != -8 || v34 )
            v21 = v34;
          v20 = (v18 - 1) & (v33 + v20);
          v69 = (_QWORD *)(v19 + 56LL * v20);
          v22 = *v69;
          if ( v15 != *v69 )
          {
            ++v33;
            v34 = v21;
            v21 = (_QWORD *)(v19 + 56LL * v20);
            continue;
          }
          break;
        }
        v23 = (_QWORD *)v69[3];
        v24 = v69 + 2;
        v21 = v69;
LABEL_18:
        v25 = v87[1];
        if ( !v23 )
        {
          v23 = v24;
          goto LABEL_53;
        }
        while ( 2 )
        {
          v28 = v23[4];
          if ( v25 < v28 )
          {
            v26 = (_QWORD *)v23[2];
            v27 = 1;
            goto LABEL_25;
          }
          if ( v25 != v28 || (unsigned __int64)v17 >= v23[5] )
          {
            v26 = (_QWORD *)v23[3];
            v27 = 0;
            if ( !v26 )
              goto LABEL_26;
LABEL_22:
            v23 = v26;
            continue;
          }
          break;
        }
        v26 = (_QWORD *)v23[2];
        v27 = 1;
LABEL_25:
        if ( v26 )
          goto LABEL_22;
LABEL_26:
        if ( v27 )
        {
          if ( v23 != (_QWORD *)v21[4] )
            goto LABEL_39;
LABEL_36:
          v30 = 1;
          if ( v24 != v23 && v25 >= v23[4] )
          {
            v30 = 0;
            if ( v25 == v23[4] )
              v30 = (unsigned __int64)v17 < v23[5];
          }
          goto LABEL_37;
        }
        v29 = v23;
        if ( v25 > v28 )
          goto LABEL_36;
LABEL_28:
        if ( v25 == v28 && (unsigned __int64)v17 > v23[5] )
        {
          v23 = v29;
          goto LABEL_35;
        }
LABEL_29:
        v13 += 8LL;
        if ( v85 != (_BYTE *)v13 )
          continue;
        break;
      }
LABEL_5:
      v88 = *(_QWORD *)(v88 + 8);
      if ( v80 == v88 )
      {
        if ( v93 != v96 )
          _libc_free((unsigned __int64)v93);
        return j___libc_free_0(i);
      }
    }
    ++v89;
    goto LABEL_92;
  }
  return j___libc_free_0(i);
}
