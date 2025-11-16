// Function: sub_1F05810
// Address: 0x1f05810
//
__int64 __fastcall sub_1F05810(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r13
  __int64 v5; // r14
  unsigned int v6; // r15d
  __int64 result; // rax
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  _WORD *v10; // rsi
  unsigned __int16 v11; // ax
  _WORD *v12; // rcx
  int v13; // edi
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 v17; // r12
  __int64 v18; // r14
  unsigned int v19; // r8d
  unsigned int v20; // esi
  __int64 v21; // rcx
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // r9
  __int64 v30; // r13
  __int64 v31; // r15
  int v32; // esi
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rbx
  unsigned __int16 v36; // r14
  __int64 *v37; // r12
  __int64 v38; // r15
  __int64 v39; // rdx
  __int16 *v40; // r13
  unsigned int v41; // edi
  unsigned int v42; // esi
  unsigned int v43; // eax
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int16 v47; // ax
  unsigned int v48; // ecx
  unsigned int v49; // esi
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdx
  unsigned int v53; // r10d
  __int64 v54; // r11
  __int64 v55; // r10
  _DWORD *v56; // rcx
  unsigned int v57; // r8d
  __int64 v58; // rdi
  unsigned int v59; // r14d
  __int64 v60; // r9
  unsigned int v61; // r9d
  unsigned int v62; // edx
  __int64 v63; // rcx
  unsigned int v64; // r10d
  unsigned int v65; // edx
  __int64 v66; // r9
  __int64 v67; // r15
  __int64 v68; // rax
  __int16 v69; // ax
  __int64 v70; // [rsp+8h] [rbp-C8h]
  __int64 v71; // [rsp+10h] [rbp-C0h]
  unsigned __int16 v72; // [rsp+1Ah] [rbp-B6h]
  unsigned int v73; // [rsp+1Ch] [rbp-B4h]
  __int64 v74; // [rsp+30h] [rbp-A0h]
  __int64 v75; // [rsp+38h] [rbp-98h]
  __int64 v76; // [rsp+40h] [rbp-90h]
  __int64 *v77; // [rsp+40h] [rbp-90h]
  int v79; // [rsp+4Ch] [rbp-84h]
  unsigned __int64 v80; // [rsp+50h] [rbp-80h] BYREF
  int v81; // [rsp+58h] [rbp-78h]
  int v82; // [rsp+5Ch] [rbp-74h]
  __int64 v83; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int64 v84; // [rsp+68h] [rbp-68h]
  char v85; // [rsp+70h] [rbp-60h]
  unsigned __int16 v86; // [rsp+78h] [rbp-58h]
  _WORD *v87; // [rsp+80h] [rbp-50h]
  int v88; // [rsp+88h] [rbp-48h]
  unsigned __int16 v89; // [rsp+90h] [rbp-40h]
  __int64 v90; // [rsp+98h] [rbp-38h]

  v3 = a1;
  v76 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(v76 + 32) + 40LL * a3;
  v6 = *(_DWORD *)(v5 + 8);
  v73 = v6;
  result = sub_1E69FD0(*(_QWORD **)(a1 + 40), v6);
  if ( (_BYTE)result )
    return result;
  v8 = *(_QWORD **)(a1 + 24);
  v79 = ((*(_BYTE *)(v5 + 3) & 0x10) != 0) + 1;
  if ( !v8 )
  {
    v84 = 0;
    v85 = 1;
    LODWORD(v83) = v6;
    v86 = 0;
    v87 = 0;
    v88 = 0;
    v89 = 0;
    v90 = 0;
    BUG();
  }
  LODWORD(v83) = v6;
  v84 = (unsigned __int64)(v8 + 1);
  v89 = 0;
  v70 = v6;
  v86 = 0;
  v87 = 0;
  v85 = 1;
  v88 = 0;
  v90 = 0;
  v71 = 24LL * v6;
  v72 = v6;
  v9 = *(_DWORD *)(v8[1] + v71 + 16);
  v10 = (_WORD *)(v8[7] + 2LL * (v9 >> 4));
  v11 = *v10 + v6 * (v9 & 0xF);
  v12 = v10 + 1;
  v86 = v11;
  v87 = v10 + 1;
  while ( 1 )
  {
    if ( !v12 )
      goto LABEL_39;
    v88 = *(_DWORD *)(v8[6] + 4LL * v86);
    v13 = (unsigned __int16)v88;
    if ( (_WORD)v88 )
      break;
LABEL_95:
    v87 = ++v12;
    v69 = *(v12 - 1);
    v86 += v69;
    if ( !v69 )
    {
      v87 = 0;
      goto LABEL_39;
    }
  }
  while ( 1 )
  {
    v14 = (unsigned __int16)v13;
    v15 = *(unsigned int *)(v8[1] + 24LL * (unsigned __int16)v13 + 8);
    v16 = v8[7];
    v89 = v13;
    v90 = v16 + 2 * v15;
    if ( v90 )
      break;
    v88 = HIWORD(v88);
    v13 = v88;
    if ( !(_WORD)v88 )
      goto LABEL_95;
  }
  v74 = v5;
  v17 = v3 + 344;
  v18 = v3;
  while ( 1 )
  {
    v19 = *(_DWORD *)(v18 + 992);
    v20 = *(unsigned __int16 *)(*(_QWORD *)(v18 + 1192) + 2 * v14);
    if ( v20 < v19 )
    {
      v21 = *(_QWORD *)(v18 + 984);
      v22 = *(unsigned __int16 *)(*(_QWORD *)(v18 + 1192) + 2 * v14);
      while ( 1 )
      {
        v23 = v21 + 24LL * v22;
        if ( v13 == *(_DWORD *)(v23 + 12) )
        {
          v24 = *(unsigned int *)(v23 + 16);
          if ( (_DWORD)v24 != -1 && *(_DWORD *)(v21 + 24 * v24 + 20) == -1 )
            break;
        }
        v22 += 0x10000;
        if ( v19 <= v22 )
          goto LABEL_30;
      }
      if ( v22 != -1 )
      {
        while ( 1 )
        {
          v25 = v20;
          v26 = v21 + 24LL * v20;
          if ( v13 == *(_DWORD *)(v26 + 12) )
          {
            v27 = *(unsigned int *)(v26 + 16);
            if ( (_DWORD)v27 != -1 && *(_DWORD *)(v21 + 24 * v27 + 20) == -1 )
              break;
          }
          v20 += 0x10000;
          if ( v19 <= v20 )
            goto LABEL_30;
        }
        if ( v20 != -1 )
        {
          do
          {
            v29 = 3 * v25;
            v30 = 24 * v25;
            v28 = v21 + 24 * v25;
            v31 = *(_QWORD *)v28;
            if ( a2 == *(_QWORD *)v28 || v31 == v17 )
              goto LABEL_24;
            v32 = v89;
            if ( v79 == 2 )
            {
              if ( (((*(_BYTE *)(v74 + 3) & 0x10) != 0) & (*(_BYTE *)(v74 + 3) >> 6)) != 0 )
              {
                if ( (unsigned int)sub_1E16810(*(_QWORD *)(v31 + 8), v89, 1, 0, 0) != -1 )
                {
                  v21 = *(_QWORD *)(v18 + 984);
                  v28 = v21 + v30;
                  goto LABEL_24;
                }
                v32 = v89;
              }
              v81 = v32;
              v82 = 0;
              v80 = a2 & 0xFFFFFFFFFFFFFFF9LL | 4;
              v82 = sub_1F4BFE0(v18 + 632, v76, a3, *(_QWORD *)(v31 + 8));
            }
            else
            {
              v81 = v89;
              v82 = 0;
              v80 = a2 & 0xFFFFFFFFFFFFFFF9LL | 2;
            }
            sub_1F01A00(v31, (__int64)&v80, 1, v21, v19, v29);
            v21 = *(_QWORD *)(v18 + 984);
            v28 = v21 + v30;
LABEL_24:
            v25 = *(unsigned int *)(v28 + 20);
          }
          while ( (_DWORD)v25 != -1 );
        }
      }
    }
LABEL_30:
    sub_1E1D5E0((__int64)&v83);
    if ( !v87 )
      break;
    v13 = v89;
    v14 = v89;
  }
  v3 = v18;
  v5 = v74;
LABEL_39:
  if ( (*(_BYTE *)(v5 + 3) & 0x10) != 0 )
  {
    sub_1F04120(v3, a2, a3);
    v34 = *(_QWORD *)(v3 + 24);
    if ( !v34 )
      BUG();
    v75 = a2;
    v35 = v5;
    v36 = v72;
    v37 = (__int64 *)(v3 + 984);
    v38 = v3;
    v39 = *(_QWORD *)(v34 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v34 + 8) + v71 + 4);
    v77 = (__int64 *)(v3 + 1216);
LABEL_42:
    v40 = (__int16 *)v39;
    while ( v40 )
    {
      v41 = *(_DWORD *)(v38 + 1224);
      v42 = v36;
      v43 = *(unsigned __int16 *)(*(_QWORD *)(v38 + 1424) + 2LL * v36);
      if ( v43 < v41 )
      {
        v44 = *(_QWORD *)(v38 + 1216);
        while ( 1 )
        {
          v45 = v44 + 24LL * v43;
          if ( v36 == *(_DWORD *)(v45 + 12) )
          {
            v46 = *(unsigned int *)(v45 + 16);
            if ( (_DWORD)v46 != -1 && *(_DWORD *)(v44 + 24 * v46 + 20) == -1 )
              break;
          }
          v43 += 0x10000;
          if ( v41 <= v43 )
            goto LABEL_52;
        }
        if ( v43 != -1 )
        {
          sub_1F03B60(v77, v36);
          v42 = v36;
        }
      }
LABEL_52:
      if ( (((*(_BYTE *)(v35 + 3) & 0x10) != 0) & (*(_BYTE *)(v35 + 3) >> 6)) == 0 )
        sub_1F03B60(v37, v42);
      v47 = *v40;
      v39 = 0;
      ++v40;
      v36 += v47;
      if ( !v47 )
        goto LABEL_42;
    }
    v33 = v38;
    if ( (((*(_BYTE *)(v35 + 3) & 0x10) != 0) & (*(_BYTE *)(v35 + 3) >> 6)) != 0 && (*(_BYTE *)(v75 + 228) & 2) != 0 )
    {
      v48 = *(_DWORD *)(v38 + 992);
      v49 = *(unsigned __int16 *)(*(_QWORD *)(v38 + 1192) + 2 * v70);
      if ( v49 < v48 )
      {
        v50 = *(_QWORD *)(v38 + 984);
        while ( 1 )
        {
          v51 = v50 + 24LL * v49;
          if ( v73 == *(_DWORD *)(v51 + 12) )
          {
            v52 = *(unsigned int *)(v51 + 16);
            if ( (_DWORD)v52 != -1 && *(_DWORD *)(v50 + 24 * v52 + 20) == -1 )
              break;
          }
          v49 += 0x10000;
          if ( v48 <= v49 )
            goto LABEL_35;
        }
        if ( v49 != -1 )
        {
          v53 = v73;
          v54 = 0xFFFFFFFFLL;
          while ( 1 )
          {
            if ( (_DWORD)v54 == -1 )
            {
              v61 = *(_DWORD *)(v33 + 992);
              v62 = *(unsigned __int16 *)(*(_QWORD *)(v33 + 1192) + 2LL * v53);
              if ( v62 < v61 )
              {
                while ( 1 )
                {
                  v63 = v50 + 24LL * v62;
                  if ( *(_DWORD *)(v63 + 12) == v53 )
                  {
                    v57 = *(_DWORD *)(v63 + 16);
                    if ( v57 != -1 )
                    {
                      v58 = 24LL * v57;
                      v56 = (_DWORD *)(v50 + v58);
                      if ( *(_DWORD *)(v50 + v58 + 20) == -1 )
                        break;
                    }
                  }
                  v62 += 0x10000;
                  if ( v61 <= v62 )
                    goto LABEL_85;
                }
              }
              else
              {
LABEL_85:
                v57 = *(_DWORD *)(v50 + 0x17FFFFFFF8LL);
                v58 = 24LL * v57;
                v56 = (_DWORD *)(v50 + v58);
              }
            }
            else
            {
              v57 = *(_DWORD *)(v50 + 24 * v54 + 16);
              v58 = 24LL * v57;
              v56 = (_DWORD *)(v50 + v58);
            }
            if ( (*(_BYTE *)(*(_QWORD *)v56 + 228LL) & 2) == 0 )
              break;
            v59 = v56[4];
            v60 = 24LL * v59;
            if ( (_DWORD *)(v50 + v60) == v56 )
            {
              v53 = v56[3];
              v54 = 0xFFFFFFFFLL;
            }
            else
            {
              v55 = (unsigned int)v56[5];
              if ( *(_DWORD *)(v50 + v60 + 20) == -1 )
              {
                *(_WORD *)(*(_QWORD *)(v33 + 1192) + 2LL * (unsigned int)v56[3]) = v55;
                *(_DWORD *)(*(_QWORD *)(v33 + 984) + 24LL * (unsigned int)v56[5] + 16) = v56[4];
                v53 = v56[3];
                v54 = (unsigned int)v56[5];
                v56 = (_DWORD *)(v58 + *(_QWORD *)(v33 + 984));
              }
              else if ( (_DWORD)v55 == -1 )
              {
                v64 = *(_DWORD *)(v33 + 992);
                v65 = *(unsigned __int16 *)(*(_QWORD *)(v33 + 1192) + 2LL * (unsigned int)v56[3]);
                if ( v65 < v64 )
                {
                  while ( 1 )
                  {
                    v66 = v50 + 24LL * v65;
                    if ( v56[3] == *(_DWORD *)(v66 + 12) )
                    {
                      v67 = *(unsigned int *)(v66 + 16);
                      if ( (_DWORD)v67 != -1 && *(_DWORD *)(v50 + 24 * v67 + 20) == -1 )
                        break;
                    }
                    v65 += 0x10000;
                    if ( v64 <= v65 )
                      goto LABEL_93;
                  }
                }
                else
                {
LABEL_93:
                  v66 = v50 + 0x17FFFFFFE8LL;
                }
                *(_DWORD *)(v66 + 16) = v59;
                *(_DWORD *)(*(_QWORD *)(v33 + 984) + 24LL * (unsigned int)v56[4] + 20) = v56[5];
                v68 = *(_QWORD *)(v33 + 984);
                v53 = v56[3];
                v54 = *(unsigned int *)(v68 + 24LL * (unsigned int)v56[4] + 20);
                v56 = (_DWORD *)(v68 + v58);
              }
              else
              {
                *(_DWORD *)(v50 + 24 * v55 + 16) = v59;
                v54 = (unsigned int)v56[5];
                *(_DWORD *)(*(_QWORD *)(v33 + 984) + v60 + 20) = v54;
                v53 = v56[3];
                v56 = (_DWORD *)(v58 + *(_QWORD *)(v33 + 984));
              }
            }
            v56[4] = -1;
            *(_DWORD *)(*(_QWORD *)(v33 + 984) + v58 + 20) = *(_DWORD *)(v33 + 1208);
            *(_DWORD *)(v33 + 1208) = v57;
            ++*(_DWORD *)(v33 + 1212);
            if ( v57 == v49 )
              break;
            v50 = *(_QWORD *)(v33 + 984);
          }
        }
      }
    }
LABEL_35:
    v83 = v75;
    v84 = __PAIR64__(v73, a3);
    return sub_1F05660((__int64)v37, (__int64)&v83);
  }
  else
  {
    v83 = a2;
    *(_BYTE *)(a2 + 228) |= 0x20u;
    v84 = __PAIR64__(v73, a3);
    result = sub_1F05660(v3 + 1216, (__int64)&v83);
    if ( *(_BYTE *)(v3 + 912) )
      *(_BYTE *)(v5 + 3) &= ~0x40u;
  }
  return result;
}
