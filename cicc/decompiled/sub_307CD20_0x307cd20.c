// Function: sub_307CD20
// Address: 0x307cd20
//
__int64 __fastcall sub_307CD20(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned __int64 v6; // rax
  __int16 v7; // dx
  int v8; // eax
  unsigned int v9; // esi
  __int64 v10; // rcx
  unsigned int v11; // edx
  int v12; // edi
  char v13; // al
  int v14; // r8d
  unsigned int v15; // esi
  __int64 v16; // rax
  unsigned int v17; // edx
  int v18; // ecx
  int v19; // r9d
  int *v20; // r9
  _DWORD *v21; // r9
  int v22; // r11d
  int v23; // ecx
  unsigned int v24; // edi
  _DWORD *v25; // r8
  int v26; // r9d
  _DWORD *v27; // rdx
  int v28; // r9d
  _DWORD *v29; // r8
  _DWORD *v30; // r8
  int v31; // eax
  int v32; // edi
  unsigned int v33; // r8d
  _DWORD *v34; // rdx
  int v35; // r9d
  int v36; // eax
  int v37; // esi
  __int64 v38; // r10
  unsigned int v39; // edi
  _DWORD *v40; // rdx
  int v41; // eax
  int v42; // r9d
  _DWORD *v43; // r8
  int v44; // edx
  __int64 v45; // r11
  unsigned int v46; // r8d
  int v47; // ecx
  int v48; // r10d
  _DWORD *v49; // r9
  int v50; // eax
  int v51; // eax
  __int64 v52; // r10
  int v53; // r9d
  unsigned int v54; // edi
  _DWORD *v55; // r11
  int v56; // ecx
  int v57; // edx
  __int64 v58; // r11
  _DWORD *v59; // r8
  int v60; // r9d
  unsigned int v61; // esi
  int v62; // r10d
  int v63; // [rsp+8h] [rbp-68h]
  int v64; // [rsp+8h] [rbp-68h]
  int v65; // [rsp+8h] [rbp-68h]
  int v66; // [rsp+8h] [rbp-68h]
  int v67; // [rsp+14h] [rbp-5Ch]
  int v68; // [rsp+14h] [rbp-5Ch]
  int v69; // [rsp+14h] [rbp-5Ch]
  int v70; // [rsp+14h] [rbp-5Ch]
  int v71; // [rsp+14h] [rbp-5Ch]
  int v72; // [rsp+14h] [rbp-5Ch]
  __int64 v73; // [rsp+18h] [rbp-58h]
  __int64 v74; // [rsp+20h] [rbp-50h]
  __int64 v75; // [rsp+28h] [rbp-48h]
  _DWORD v76[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v73 = a1 + 56;
  result = *(_QWORD *)a1 + 320LL;
  v74 = result;
  v75 = *(_QWORD *)(*(_QWORD *)a1 + 328LL);
  if ( v75 != result )
  {
    while ( 1 )
    {
      v3 = *(_QWORD *)(v75 + 56);
      if ( v3 != v75 + 48 )
        break;
LABEL_23:
      result = *(_QWORD *)(v75 + 8);
      v75 = result;
      if ( v74 == result )
        return result;
    }
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 + 32);
      v5 = v4 + 40LL * (*(_DWORD *)(v3 + 40) & 0xFFFFFF);
      if ( v5 != v4 )
        break;
LABEL_21:
      if ( (*(_BYTE *)v3 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v3 + 44) & 8) != 0 )
          v3 = *(_QWORD *)(v3 + 8);
      }
      v3 = *(_QWORD *)(v3 + 8);
      if ( v75 + 48 == v3 )
        goto LABEL_23;
    }
    while ( 1 )
    {
      if ( *(_BYTE *)v4 )
        goto LABEL_10;
      v13 = *(_BYTE *)(v4 + 3);
      if ( (v13 & 0x20) != 0 )
        goto LABEL_10;
      v14 = *(_DWORD *)(v4 + 8);
      if ( v14 >= 0 )
        goto LABEL_10;
      if ( (v13 & 0x10) == 0 )
        goto LABEL_5;
      if ( *(_WORD *)(v3 + 68) != 68 && *(_WORD *)(v3 + 68) )
        goto LABEL_10;
      v15 = *(_DWORD *)(a1 + 80);
      v16 = *(_QWORD *)(a1 + 64);
      v76[0] = *(_DWORD *)(v4 + 8);
      if ( !v15 )
        goto LABEL_30;
      v17 = (v15 - 1) & (37 * v14);
      v18 = *(_DWORD *)(v16 + 8LL * v17);
      if ( v14 != v18 )
        break;
LABEL_19:
      if ( (*(_BYTE *)(v4 + 3) & 0x10) != 0 )
        goto LABEL_10;
      v14 = *(_DWORD *)(v4 + 8);
LABEL_5:
      v6 = sub_2EBEE10(*(_QWORD *)(a1 + 192), v14);
      if ( !v6 )
        goto LABEL_10;
      v7 = *(_WORD *)(v6 + 68);
      if ( v7 == 10 || *(_QWORD *)(v6 + 24) == v75 && v7 != 68 && v7 )
        goto LABEL_10;
      v8 = *(_DWORD *)(v4 + 8);
      v9 = *(_DWORD *)(a1 + 80);
      v10 = *(_QWORD *)(a1 + 64);
      v76[0] = v8;
      if ( !v9 )
        goto LABEL_42;
      v11 = (v9 - 1) & (37 * v8);
      v12 = *(_DWORD *)(v10 + 8LL * v11);
      if ( v8 != v12 )
      {
        v28 = 1;
        while ( v12 != -1 )
        {
          v11 = (v9 - 1) & (v28 + v11);
          v12 = *(_DWORD *)(v10 + 8LL * v11);
          if ( v8 == v12 )
            goto LABEL_10;
          ++v28;
        }
LABEL_42:
        v29 = *(_DWORD **)(a1 + 96);
        if ( v29 == *(_DWORD **)(a1 + 104) )
        {
          sub_B8BBF0(a1 + 88, *(_BYTE **)(a1 + 96), v76);
          v9 = *(_DWORD *)(a1 + 80);
          v10 = *(_QWORD *)(a1 + 64);
          v31 = ((__int64)(*(_QWORD *)(a1 + 96) - *(_QWORD *)(a1 + 88)) >> 2) - 1;
          if ( v9 )
            goto LABEL_46;
        }
        else
        {
          if ( v29 )
          {
            *v29 = v8;
            v29 = *(_DWORD **)(a1 + 96);
            v10 = *(_QWORD *)(a1 + 64);
            v9 = *(_DWORD *)(a1 + 80);
          }
          v30 = v29 + 1;
          *(_QWORD *)(a1 + 96) = v30;
          v31 = (((__int64)v30 - *(_QWORD *)(a1 + 88)) >> 2) - 1;
          if ( v9 )
          {
LABEL_46:
            v32 = v76[0];
            v33 = (v9 - 1) & (37 * v76[0]);
            v34 = (_DWORD *)(v10 + 8LL * v33);
            v35 = *v34;
            if ( v76[0] == *v34 )
            {
LABEL_47:
              v34[1] = v31;
              goto LABEL_10;
            }
            v71 = 1;
            v55 = 0;
            while ( v35 != -1 )
            {
              if ( v35 == -2 && !v55 )
                v55 = v34;
              v33 = (v9 - 1) & (v71 + v33);
              v34 = (_DWORD *)(v10 + 8LL * v33);
              v35 = *v34;
              if ( v76[0] == *v34 )
                goto LABEL_47;
              ++v71;
            }
            v56 = *(_DWORD *)(a1 + 72);
            if ( v55 )
              v34 = v55;
            ++*(_QWORD *)(a1 + 56);
            v47 = v56 + 1;
            if ( 4 * v47 < 3 * v9 )
            {
              if ( v9 - (v47 + *(_DWORD *)(a1 + 76)) <= v9 >> 3 )
              {
                v66 = v31;
                sub_2E518D0(v73, v9);
                v57 = *(_DWORD *)(a1 + 80);
                if ( !v57 )
                {
LABEL_128:
                  ++*(_DWORD *)(a1 + 72);
                  BUG();
                }
                v32 = v76[0];
                v58 = *(_QWORD *)(a1 + 64);
                v59 = 0;
                v72 = v57 - 1;
                v60 = 1;
                v61 = (v57 - 1) & (37 * v76[0]);
                v34 = (_DWORD *)(v58 + 8LL * v61);
                v47 = *(_DWORD *)(a1 + 72) + 1;
                v31 = v66;
                v62 = *v34;
                if ( *v34 != v76[0] )
                {
                  while ( v62 != -1 )
                  {
                    if ( v62 == -2 && !v59 )
                      v59 = v34;
                    v61 = v72 & (v60 + v61);
                    v34 = (_DWORD *)(v58 + 8LL * v61);
                    v62 = *v34;
                    if ( v76[0] == *v34 )
                      goto LABEL_91;
                    ++v60;
                  }
                  if ( v59 )
                    v34 = v59;
                }
              }
              goto LABEL_91;
            }
LABEL_59:
            v64 = v31;
            sub_2E518D0(v73, 2 * v9);
            v44 = *(_DWORD *)(a1 + 80);
            if ( !v44 )
              goto LABEL_128;
            v45 = *(_QWORD *)(a1 + 64);
            v68 = v44 - 1;
            v46 = (v44 - 1) & (37 * v76[0]);
            v34 = (_DWORD *)(v45 + 8LL * v46);
            v47 = *(_DWORD *)(a1 + 72) + 1;
            v31 = v64;
            v32 = *v34;
            if ( v76[0] != *v34 )
            {
              v48 = 1;
              v49 = 0;
              while ( v32 != -1 )
              {
                if ( !v49 && v32 == -2 )
                  v49 = v34;
                v46 = v68 & (v48 + v46);
                v34 = (_DWORD *)(v45 + 8LL * v46);
                v32 = *v34;
                if ( v76[0] == *v34 )
                  goto LABEL_91;
                ++v48;
              }
              v32 = v76[0];
              if ( v49 )
                v34 = v49;
            }
LABEL_91:
            *(_DWORD *)(a1 + 72) = v47;
            if ( *v34 != -1 )
              --*(_DWORD *)(a1 + 76);
            *v34 = v32;
            v34[1] = 0;
            goto LABEL_47;
          }
        }
        ++*(_QWORD *)(a1 + 56);
        goto LABEL_59;
      }
LABEL_10:
      v4 += 40;
      if ( v5 == v4 )
        goto LABEL_21;
    }
    v19 = 1;
    while ( v18 != -1 )
    {
      v17 = (v15 - 1) & (v19 + v17);
      v18 = *(_DWORD *)(v16 + 8LL * v17);
      if ( v14 == v18 )
        goto LABEL_19;
      ++v19;
    }
LABEL_30:
    v20 = *(int **)(a1 + 96);
    if ( v20 == *(int **)(a1 + 104) )
    {
      sub_B8BBF0(a1 + 88, *(_BYTE **)(a1 + 96), v76);
      v15 = *(_DWORD *)(a1 + 80);
      v16 = *(_QWORD *)(a1 + 64);
      v22 = ((__int64)(*(_QWORD *)(a1 + 96) - *(_QWORD *)(a1 + 88)) >> 2) - 1;
      if ( v15 )
        goto LABEL_34;
    }
    else
    {
      if ( v20 )
      {
        *v20 = v14;
        v20 = *(int **)(a1 + 96);
        v16 = *(_QWORD *)(a1 + 64);
        v15 = *(_DWORD *)(a1 + 80);
      }
      v21 = v20 + 1;
      *(_QWORD *)(a1 + 96) = v21;
      v22 = (((__int64)v21 - *(_QWORD *)(a1 + 88)) >> 2) - 1;
      if ( v15 )
      {
LABEL_34:
        v23 = v76[0];
        v24 = (v15 - 1) & (37 * v76[0]);
        v25 = (_DWORD *)(v16 + 8LL * v24);
        v26 = *v25;
        if ( v76[0] == *v25 )
        {
LABEL_35:
          v27 = v25 + 1;
LABEL_36:
          *v27 = v22;
          goto LABEL_19;
        }
        v69 = 1;
        v40 = 0;
        while ( v26 != -1 )
        {
          if ( !v40 && v26 == -2 )
            v40 = v25;
          v24 = (v15 - 1) & (v69 + v24);
          v25 = (_DWORD *)(v16 + 8LL * v24);
          v26 = *v25;
          if ( v76[0] == *v25 )
            goto LABEL_35;
          ++v69;
        }
        v50 = *(_DWORD *)(a1 + 72);
        if ( !v40 )
          v40 = v25;
        ++*(_QWORD *)(a1 + 56);
        v41 = v50 + 1;
        if ( 4 * v41 < 3 * v15 )
        {
          if ( v15 - (v41 + *(_DWORD *)(a1 + 76)) > v15 >> 3 )
            goto LABEL_72;
          v65 = v22;
          sub_2E518D0(v73, v15);
          v51 = *(_DWORD *)(a1 + 80);
          if ( !v51 )
          {
LABEL_127:
            ++*(_DWORD *)(a1 + 72);
            BUG();
          }
          v37 = v76[0];
          v52 = *(_QWORD *)(a1 + 64);
          v43 = 0;
          v70 = v51 - 1;
          v22 = v65;
          v53 = 1;
          v54 = (v51 - 1) & (37 * v76[0]);
          v40 = (_DWORD *)(v52 + 8LL * v54);
          v23 = *v40;
          v41 = *(_DWORD *)(a1 + 72) + 1;
          if ( v76[0] == *v40 )
            goto LABEL_72;
          while ( v23 != -1 )
          {
            if ( !v43 && v23 == -2 )
              v43 = v40;
            v54 = v70 & (v53 + v54);
            v40 = (_DWORD *)(v52 + 8LL * v54);
            v23 = *v40;
            if ( v76[0] == *v40 )
              goto LABEL_72;
            ++v53;
          }
          goto LABEL_54;
        }
LABEL_50:
        v63 = v22;
        sub_2E518D0(v73, 2 * v15);
        v36 = *(_DWORD *)(a1 + 80);
        if ( !v36 )
          goto LABEL_127;
        v37 = v76[0];
        v38 = *(_QWORD *)(a1 + 64);
        v67 = v36 - 1;
        v22 = v63;
        v39 = (v36 - 1) & (37 * v76[0]);
        v40 = (_DWORD *)(v38 + 8LL * v39);
        v23 = *v40;
        v41 = *(_DWORD *)(a1 + 72) + 1;
        if ( v76[0] == *v40 )
          goto LABEL_72;
        v42 = 1;
        v43 = 0;
        while ( v23 != -1 )
        {
          if ( v23 == -2 && !v43 )
            v43 = v40;
          v39 = v67 & (v42 + v39);
          v40 = (_DWORD *)(v38 + 8LL * v39);
          v23 = *v40;
          if ( v76[0] == *v40 )
            goto LABEL_72;
          ++v42;
        }
LABEL_54:
        v23 = v37;
        if ( v43 )
          v40 = v43;
LABEL_72:
        *(_DWORD *)(a1 + 72) = v41;
        if ( *v40 != -1 )
          --*(_DWORD *)(a1 + 76);
        *v40 = v23;
        v27 = v40 + 1;
        *v27 = 0;
        goto LABEL_36;
      }
    }
    ++*(_QWORD *)(a1 + 56);
    goto LABEL_50;
  }
  return result;
}
