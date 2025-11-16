// Function: sub_2060560
// Address: 0x2060560
//
unsigned __int64 __fastcall sub_2060560(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  int v8; // eax
  __int64 v9; // r13
  unsigned int v10; // ebx
  unsigned __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // r12
  __int64 v15; // r8
  unsigned int v16; // edx
  unsigned int v17; // edi
  __int64 *v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // edx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // r9
  unsigned int v25; // esi
  __int64 *v26; // r13
  __int64 v27; // rbx
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  __int64 v32; // rax
  __int64 *v33; // r10
  _QWORD *i; // rax
  __int64 *j; // rax
  __int64 v36; // r11
  int v37; // esi
  int v38; // esi
  __int64 v39; // r8
  unsigned int v40; // ecx
  _QWORD *v41; // rdx
  __int64 v42; // rdi
  int v43; // esi
  int v44; // ecx
  int v45; // esi
  unsigned int v46; // edi
  __int64 v47; // rdi
  int v48; // r10d
  int v49; // edi
  int v50; // esi
  int v51; // esi
  int v52; // r11d
  __int64 *v53; // r10
  __int64 v54; // rdx
  __int64 v55; // rdi
  __int64 v56; // rcx
  _QWORD *k; // rcx
  _QWORD *v58; // rax
  int v59; // r8d
  int v60; // r9d
  unsigned int v61; // edx
  _QWORD *v62; // rbx
  _QWORD *v63; // rbx
  int v64; // r8d
  int v65; // r9d
  unsigned int v66; // eax
  __int64 v67; // rcx
  __int64 v68; // rdx
  __int64 v69; // rax
  _QWORD *v70; // r9
  int v71; // r11d
  __int64 *v72; // [rsp+8h] [rbp-68h]
  unsigned int v73; // [rsp+8h] [rbp-68h]
  int v74; // [rsp+8h] [rbp-68h]
  __int64 v75; // [rsp+18h] [rbp-58h]
  unsigned int v77; // [rsp+28h] [rbp-48h]
  unsigned int v78; // [rsp+2Ch] [rbp-44h]
  __int64 *v79; // [rsp+30h] [rbp-40h]
  __int64 v80[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = *(_QWORD *)a1;
  v80[0] = a2;
  v7 = sub_15E38F0(v6);
  v8 = sub_14DD7D0(v7);
  v9 = a2;
  v10 = v8 - 9;
  v78 = v8 - 7;
  result = a1 + 48;
  v77 = v10;
LABEL_2:
  if ( v9 )
  {
    while ( 1 )
    {
      v75 = sub_157ED20(v9);
      result = *(unsigned __int8 *)(v75 + 16);
      if ( (_BYTE)result == 88 )
        break;
      if ( (_BYTE)result == 73 )
      {
        v63 = sub_1FE1990(a1 + 48, v80);
        v66 = *(_DWORD *)(a4 + 8);
        if ( v66 >= *(_DWORD *)(a4 + 12) )
        {
          sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v64, v65);
          v66 = *(_DWORD *)(a4 + 8);
        }
        v67 = *(_QWORD *)a4;
        v68 = *(_QWORD *)a4 + 16LL * v66;
        if ( v68 )
        {
          *(_QWORD *)v68 = v63[1];
          *(_DWORD *)(v68 + 8) = a3;
          v67 = *(_QWORD *)a4;
          v66 = *(_DWORD *)(a4 + 8);
        }
        v69 = v66 + 1;
        *(_DWORD *)(a4 + 8) = v69;
        *(_BYTE *)(*(_QWORD *)(v67 + 16 * v69 - 16) + 182LL) = 1;
        result = *(_QWORD *)(*(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8) - 16);
        *(_BYTE *)(result + 183) = 1;
        return result;
      }
      v9 = v80[0];
      if ( (_BYTE)result != 34 )
        goto LABEL_2;
      v12 = 24LL * (*(_DWORD *)(v75 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v75 + 23) & 0x40) != 0 )
      {
        v13 = *(_QWORD *)(v75 - 8);
        v79 = (__int64 *)(v13 + v12);
      }
      else
      {
        v79 = (__int64 *)v75;
        v13 = v75 - v12;
      }
      if ( (*(_BYTE *)(v75 + 18) & 1) == 0 )
      {
        result = v13 + 24;
        if ( (__int64 *)result == v79 )
        {
          v9 = 0;
          goto LABEL_49;
        }
LABEL_10:
        v14 = (__int64 *)result;
        while ( 2 )
        {
          v23 = sub_1523720(*v14);
          v25 = *(_DWORD *)(a1 + 72);
          v26 = *(__int64 **)(a1 + 56);
          v27 = v23;
          if ( !v25 )
          {
            ++*(_QWORD *)(a1 + 48);
            goto LABEL_22;
          }
          LODWORD(v15) = v25 - 1;
          v16 = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
          v17 = (v25 - 1) & v16;
          v18 = &v26[2 * v17];
          v19 = *v18;
          if ( v27 == *v18 )
          {
LABEL_12:
            v20 = *(_DWORD *)(a4 + 8);
            if ( v20 < *(_DWORD *)(a4 + 12) )
              goto LABEL_13;
LABEL_42:
            v72 = v18;
            sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v15, (int)v24);
            v20 = *(_DWORD *)(a4 + 8);
            v18 = v72;
LABEL_13:
            v21 = *(_QWORD *)a4 + 16LL * v20;
            if ( v21 )
            {
              *(_QWORD *)v21 = v18[1];
              *(_DWORD *)(v21 + 8) = a3;
              v20 = *(_DWORD *)(a4 + 8);
            }
            v22 = v20 + 1;
            *(_DWORD *)(a4 + 8) = v22;
            if ( v77 <= 1 )
              *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a4 + 16 * v22 - 16) + 183LL) = 1;
            if ( v78 > 1 )
              *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8) - 16) + 182LL) = 1;
            v14 += 3;
            if ( v79 == v14 )
            {
              result = v75;
              v9 = 0;
              if ( (*(_BYTE *)(v75 + 18) & 1) != 0 )
              {
                if ( (*(_BYTE *)(v75 + 23) & 0x40) != 0 )
                  goto LABEL_45;
LABEL_65:
                result = v75 - 24LL * (*(_DWORD *)(v75 + 20) & 0xFFFFFFF);
                goto LABEL_46;
              }
              goto LABEL_49;
            }
            continue;
          }
          break;
        }
        v48 = 1;
        v24 = 0;
        while ( v19 != -8 )
        {
          if ( !v24 && v19 == -16 )
            v24 = v18;
          v17 = v15 & (v48 + v17);
          v18 = &v26[2 * v17];
          v19 = *v18;
          if ( v27 == *v18 )
            goto LABEL_12;
          ++v48;
        }
        v49 = *(_DWORD *)(a1 + 64);
        if ( v24 )
          v18 = v24;
        ++*(_QWORD *)(a1 + 48);
        v44 = v49 + 1;
        if ( 4 * (v49 + 1) >= 3 * v25 )
        {
LABEL_22:
          v28 = 2 * v25 - 1;
          v29 = (((((((((v28 | (v28 >> 1)) >> 2) | v28 | (v28 >> 1)) >> 4) | ((v28 | (v28 >> 1)) >> 2)
                                                                           | v28
                                                                           | (v28 >> 1)) >> 8)
                 | ((((v28 | (v28 >> 1)) >> 2) | v28 | (v28 >> 1)) >> 4)
                 | ((v28 | (v28 >> 1)) >> 2)
                 | v28
                 | (v28 >> 1)) >> 16)
               | ((((((v28 | (v28 >> 1)) >> 2) | v28 | (v28 >> 1)) >> 4) | ((v28 | (v28 >> 1)) >> 2) | v28 | (v28 >> 1)) >> 8)
               | ((((v28 | (v28 >> 1)) >> 2) | v28 | (v28 >> 1)) >> 4)
               | ((v28 | (v28 >> 1)) >> 2)
               | v28
               | (v28 >> 1))
              + 1;
          if ( (unsigned int)v29 < 0x40 )
            LODWORD(v29) = 64;
          *(_DWORD *)(a1 + 72) = v29;
          v30 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v29);
          *(_QWORD *)(a1 + 56) = v30;
          v31 = v30;
          if ( v26 )
          {
            v32 = *(unsigned int *)(a1 + 72);
            *(_QWORD *)(a1 + 64) = 0;
            v33 = &v26[2 * v25];
            for ( i = &v31[2 * v32]; i != v31; v31 += 2 )
            {
              if ( v31 )
                *v31 = -8;
            }
            for ( j = v26; v33 != j; j += 2 )
            {
              v36 = *j;
              if ( *j != -16 && v36 != -8 )
              {
                v37 = *(_DWORD *)(a1 + 72);
                if ( !v37 )
                {
                  MEMORY[0] = *j;
                  BUG();
                }
                v38 = v37 - 1;
                v39 = *(_QWORD *)(a1 + 56);
                v40 = v38 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
                v41 = (_QWORD *)(v39 + 16LL * v40);
                v42 = *v41;
                if ( v36 != *v41 )
                {
                  v74 = 1;
                  v70 = 0;
                  while ( v42 != -8 )
                  {
                    if ( !v70 && v42 == -16 )
                      v70 = v41;
                    v40 = v38 & (v74 + v40);
                    v41 = (_QWORD *)(v39 + 16LL * v40);
                    v42 = *v41;
                    if ( v36 == *v41 )
                      goto LABEL_34;
                    ++v74;
                  }
                  if ( v70 )
                    v41 = v70;
                }
LABEL_34:
                *v41 = v36;
                v41[1] = j[1];
                ++*(_DWORD *)(a1 + 64);
              }
            }
            j___libc_free_0(v26);
            v31 = *(_QWORD **)(a1 + 56);
            v43 = *(_DWORD *)(a1 + 72);
            v44 = *(_DWORD *)(a1 + 64) + 1;
          }
          else
          {
            v56 = *(unsigned int *)(a1 + 72);
            *(_QWORD *)(a1 + 64) = 0;
            v43 = v56;
            for ( k = &v30[2 * v56]; k != v30; v30 += 2 )
            {
              if ( v30 )
                *v30 = -8;
            }
            v44 = 1;
          }
          if ( !v43 )
          {
LABEL_110:
            ++*(_DWORD *)(a1 + 64);
            BUG();
          }
          v45 = v43 - 1;
          v46 = v45 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v18 = &v31[2 * v46];
          v15 = *v18;
          if ( v27 != *v18 )
          {
            v71 = 1;
            v53 = 0;
            while ( v15 != -8 )
            {
              if ( v15 == -16 && !v53 )
                v53 = v18;
              LODWORD(v24) = v71 + 1;
              v46 = v45 & (v71 + v46);
              v18 = &v31[2 * v46];
              v15 = *v18;
              if ( v27 == *v18 )
                goto LABEL_39;
              ++v71;
            }
            goto LABEL_60;
          }
        }
        else
        {
          LODWORD(v15) = v25 >> 3;
          if ( v25 - *(_DWORD *)(a1 + 68) - v44 <= v25 >> 3 )
          {
            v73 = v16;
            sub_1D52F30(a1 + 48, v25);
            v50 = *(_DWORD *)(a1 + 72);
            if ( !v50 )
              goto LABEL_110;
            v51 = v50 - 1;
            v15 = *(_QWORD *)(a1 + 56);
            v52 = 1;
            v53 = 0;
            LODWORD(v54) = v51 & v73;
            v44 = *(_DWORD *)(a1 + 64) + 1;
            v18 = (__int64 *)(v15 + 16LL * (v51 & v73));
            v55 = *v18;
            if ( v27 != *v18 )
            {
              while ( v55 != -8 )
              {
                if ( !v53 && v55 == -16 )
                  v53 = v18;
                LODWORD(v24) = v52 + 1;
                v54 = v51 & (unsigned int)(v54 + v52);
                v18 = (__int64 *)(v15 + 16 * v54);
                v55 = *v18;
                if ( v27 == *v18 )
                  goto LABEL_39;
                ++v52;
              }
LABEL_60:
              if ( v53 )
                v18 = v53;
            }
          }
        }
LABEL_39:
        *(_DWORD *)(a1 + 64) = v44;
        if ( *v18 != -8 )
          --*(_DWORD *)(a1 + 68);
        *v18 = v27;
        v18[1] = 0;
        v20 = *(_DWORD *)(a4 + 8);
        if ( v20 >= *(_DWORD *)(a4 + 12) )
          goto LABEL_42;
        goto LABEL_13;
      }
      result = v13 + 48;
      if ( (__int64 *)result != v79 )
        goto LABEL_10;
      if ( (*(_BYTE *)(v75 + 23) & 0x40) == 0 )
        goto LABEL_65;
LABEL_45:
      result = *(_QWORD *)(v75 - 8);
LABEL_46:
      v9 = *(_QWORD *)(result + 24);
      if ( v9 )
      {
        v47 = *(_QWORD *)(a1 + 32);
        if ( v47 )
        {
          result = (a3 * (unsigned __int64)(unsigned int)sub_13774B0(v47, v80[0], *(_QWORD *)(result + 24)) + 0x40000000) >> 31;
          a3 = result;
        }
      }
LABEL_49:
      v80[0] = v9;
      if ( !v9 )
        return result;
    }
    v58 = sub_1FE1990(a1 + 48, v80);
    v61 = *(_DWORD *)(a4 + 8);
    v62 = v58;
    if ( v61 >= *(_DWORD *)(a4 + 12) )
    {
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v59, v60);
      v61 = *(_DWORD *)(a4 + 8);
    }
    result = *(_QWORD *)a4 + 16LL * v61;
    if ( result )
    {
      *(_QWORD *)result = v62[1];
      *(_DWORD *)(result + 8) = a3;
      v61 = *(_DWORD *)(a4 + 8);
    }
    *(_DWORD *)(a4 + 8) = v61 + 1;
  }
  return result;
}
