// Function: sub_269AE30
// Address: 0x269ae30
//
void __fastcall sub_269AE30(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned int v28; // r15d
  __int64 v29; // r14
  int v30; // eax
  __int64 v31; // rcx
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  __int64 v34; // rax
  __int64 *v35; // r14
  __int64 v36; // r9
  unsigned int v37; // eax
  _QWORD *v38; // rdx
  __int64 v39; // rdi
  int v40; // eax
  __int64 *v41; // rax
  int v42; // eax
  int v43; // edx
  unsigned int v44; // eax
  __int64 v45; // rdi
  int v46; // r9d
  __int64 v47; // r9
  __int64 v48; // r8
  _QWORD *v49; // rdi
  __int64 v50; // rdx
  _QWORD *v51; // rcx
  __int64 v52; // r15
  int v53; // eax
  _QWORD *v54; // rax
  __int64 v55; // rax
  __int64 v56; // r14
  __int64 v57; // rdx
  int v58; // r11d
  _QWORD *v59; // r8
  int v60; // eax
  __int64 v61; // [rsp-68h] [rbp-68h]
  __int64 v62; // [rsp-60h] [rbp-60h]
  _QWORD *v63; // [rsp-58h] [rbp-58h]
  unsigned int v64; // [rsp-50h] [rbp-50h]
  __int64 *v65; // [rsp-50h] [rbp-50h]
  __int64 v66; // [rsp-48h] [rbp-48h] BYREF
  __int64 v67[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( a2 )
  {
    v3 = a2;
    do
    {
      v4 = *(_QWORD *)(v3 + 24);
      if ( *(_BYTE *)v4 != 85 || v3 < v4 - 32 * (unsigned __int64)(*(_DWORD *)(v4 + 4) & 0x7FFFFFF) )
        goto LABEL_3;
      if ( *(char *)(v4 + 7) < 0 )
      {
        v5 = sub_BD2BC0(*(_QWORD *)(v3 + 24));
        v7 = v5 + v6;
        if ( *(char *)(v4 + 7) >= 0 )
        {
          if ( (unsigned int)(v7 >> 4) )
LABEL_116:
            BUG();
        }
        else if ( (unsigned int)((v7 - sub_BD2BC0(v4)) >> 4) )
        {
          if ( *(char *)(v4 + 7) >= 0 )
            goto LABEL_116;
          v8 = *(_DWORD *)(sub_BD2BC0(v4) + 8);
          if ( *(char *)(v4 + 7) >= 0 )
            BUG();
          v9 = sub_BD2BC0(v4);
          v11 = -32 - 32LL * (unsigned int)(*(_DWORD *)(v9 + v10 - 4) - v8);
          goto LABEL_15;
        }
      }
      v11 = -32;
LABEL_15:
      if ( v3 >= v4 + v11 )
        goto LABEL_3;
      v12 = *(_QWORD *)(v4 - 32);
      if ( !v12 )
        goto LABEL_3;
      if ( *(_BYTE *)v12 )
        goto LABEL_3;
      if ( *(_QWORD *)(v12 + 24) != *(_QWORD *)(v4 + 80) )
        goto LABEL_3;
      v63 = *(_QWORD **)a1;
      v64 = sub_BD2910(v3);
      if ( (*(_BYTE *)(v12 + 32) & 0xFu) - 7 > 1 )
        goto LABEL_3;
      if ( !*(_QWORD *)(v12 + 16) )
        goto LABEL_38;
      v62 = v12;
      v13 = *(_QWORD *)(v12 + 16);
      do
      {
        a2 = 0;
        v14 = sub_266E210(v13, 0);
        if ( !v14 )
          goto LABEL_3;
        a2 = *(_DWORD *)(v14 + 4) & 0x7FFFFFF;
        v15 = *(_QWORD *)(v14 + 32 * (v64 - a2));
        v67[0] = v15;
        if ( v4 != v14 )
        {
          v16 = *v63;
          if ( *(_DWORD *)(*v63 + 16LL) )
          {
            a2 = *(_QWORD *)(v16 + 8);
            v42 = *(_DWORD *)(v16 + 24);
            if ( v42 )
            {
              v43 = v42 - 1;
              v44 = (v42 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
              v45 = *(_QWORD *)(a2 + 8LL * v44);
              if ( v15 == v45 )
                goto LABEL_36;
              v46 = 1;
              while ( v45 != -4096 )
              {
                v44 = v43 & (v46 + v44);
                v45 = *(_QWORD *)(a2 + 8LL * v44);
                if ( v15 == v45 )
                  goto LABEL_36;
                ++v46;
              }
            }
          }
          else
          {
            v17 = *(_QWORD **)(v16 + 32);
            a2 = (__int64)&v17[*(unsigned int *)(v16 + 40)];
            if ( (_QWORD *)a2 != sub_266E410(v17, a2, v67) )
              goto LABEL_36;
          }
          if ( *(_BYTE *)v15 != 85 )
            goto LABEL_3;
          v61 = *(_QWORD *)(v63[1] + 72LL);
          if ( *(char *)(v15 + 7) < 0 )
          {
            v18 = sub_BD2BC0(v15);
            a2 = v18 + v19;
            if ( *(char *)(v15 + 7) >= 0 )
            {
              v20 = a2 >> 4;
            }
            else
            {
              a2 = (v18 + v19 - sub_BD2BC0(v15)) >> 4;
              LODWORD(v20) = a2;
            }
            if ( (_DWORD)v20 )
              goto LABEL_3;
          }
          v21 = *(_QWORD *)(v61 + 4432);
          if ( !v21 )
            goto LABEL_3;
          v22 = *(_QWORD *)(v15 - 32);
          if ( !v22 || *(_BYTE *)v22 || *(_QWORD *)(v22 + 24) != *(_QWORD *)(v15 + 80) || v21 != v22 )
            goto LABEL_3;
        }
LABEL_36:
        v13 = *(_QWORD *)(v13 + 8);
      }
      while ( v13 );
      v12 = v62;
LABEL_38:
      v23 = *(_QWORD *)(a1 + 8);
      v28 = sub_BD2910(v3);
      if ( (*(_BYTE *)(v12 + 2) & 1) != 0 )
        sub_B2C6D0(v12, a2, v24, v25);
      v29 = *(_QWORD *)(v12 + 96) + 40LL * v28;
      v30 = *(_DWORD *)(v23 + 16);
      v66 = v29;
      if ( v30 )
      {
        a2 = *(unsigned int *)(v23 + 24);
        if ( (_DWORD)a2 )
        {
          v47 = *(_QWORD *)(v23 + 8);
          v48 = 1;
          v49 = 0;
          LODWORD(v50) = (a2 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
          v51 = (_QWORD *)(v47 + 8LL * (unsigned int)v50);
          v52 = *v51;
          if ( v29 == *v51 )
            goto LABEL_3;
          while ( v52 != -4096 )
          {
            if ( v52 == -8192 && !v49 )
              v49 = v51;
            v50 = ((_DWORD)a2 - 1) & (unsigned int)(v50 + v48);
            v51 = (_QWORD *)(v47 + 8 * v50);
            v52 = *v51;
            if ( v29 == *v51 )
              goto LABEL_3;
            v48 = (unsigned int)(v48 + 1);
          }
          if ( v49 )
            v51 = v49;
          v53 = v30 + 1;
          v67[0] = (__int64)v51;
          ++*(_QWORD *)v23;
          if ( 4 * v53 < (unsigned int)(3 * a2) )
          {
            if ( (int)a2 - *(_DWORD *)(v23 + 20) - v53 > (unsigned int)a2 >> 3 )
            {
LABEL_76:
              *(_DWORD *)(v23 + 16) = v53;
              v54 = (_QWORD *)v67[0];
              if ( *(_QWORD *)v67[0] != -4096 )
                --*(_DWORD *)(v23 + 20);
              *v54 = v66;
              v55 = *(unsigned int *)(v23 + 40);
              v56 = v66;
              if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 44) )
              {
                a2 = v23 + 48;
                sub_C8D5F0(v23 + 32, (const void *)(v23 + 48), v55 + 1, 8u, v48, v47);
                v55 = *(unsigned int *)(v23 + 40);
              }
              *(_QWORD *)(*(_QWORD *)(v23 + 32) + 8 * v55) = v56;
              ++*(_DWORD *)(v23 + 40);
              goto LABEL_3;
            }
LABEL_102:
            sub_CE2A30(v23, a2);
            a2 = (__int64)&v66;
            sub_DA5B20(v23, &v66, v67);
            v53 = *(_DWORD *)(v23 + 16) + 1;
            goto LABEL_76;
          }
        }
        else
        {
          v67[0] = 0;
          ++*(_QWORD *)v23;
        }
        LODWORD(a2) = 2 * a2;
        goto LABEL_102;
      }
      v31 = *(unsigned int *)(v23 + 40);
      v32 = *(_QWORD **)(v23 + 32);
      a2 = (__int64)&v32[v31];
      if ( !((8 * v31) >> 5) )
      {
LABEL_81:
        v57 = a2 - (_QWORD)v32;
        if ( a2 - (_QWORD)v32 != 16 )
        {
          if ( v57 != 24 )
          {
            if ( v57 != 8 )
              goto LABEL_49;
            goto LABEL_84;
          }
          if ( v29 == *v32 )
            goto LABEL_48;
          ++v32;
        }
        if ( v29 == *v32 )
          goto LABEL_48;
        ++v32;
LABEL_84:
        if ( v29 == *v32 )
          goto LABEL_48;
        goto LABEL_49;
      }
      v33 = &v32[4 * ((8 * v31) >> 5)];
      while ( v29 != *v32 )
      {
        if ( v29 == v32[1] )
        {
          ++v32;
          break;
        }
        if ( v29 == v32[2] )
        {
          v32 += 2;
          break;
        }
        if ( v29 == v32[3] )
        {
          v32 += 3;
          break;
        }
        v32 += 4;
        if ( v33 == v32 )
          goto LABEL_81;
      }
LABEL_48:
      if ( (_QWORD *)a2 == v32 )
      {
LABEL_49:
        if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 44) )
        {
          sub_C8D5F0(v23 + 32, (const void *)(v23 + 48), v31 + 1, 8u, v26, v27);
          a2 = *(_QWORD *)(v23 + 32) + 8LL * *(unsigned int *)(v23 + 40);
        }
        *(_QWORD *)a2 = v29;
        v34 = (unsigned int)(*(_DWORD *)(v23 + 40) + 1);
        *(_DWORD *)(v23 + 40) = v34;
        if ( (unsigned int)v34 > 0x10 )
        {
          v35 = *(__int64 **)(v23 + 32);
          v65 = &v35[v34];
          while ( 1 )
          {
            a2 = *(unsigned int *)(v23 + 24);
            if ( !(_DWORD)a2 )
              break;
            v36 = *(_QWORD *)(v23 + 8);
            v37 = (a2 - 1) & (((unsigned int)*v35 >> 9) ^ ((unsigned int)*v35 >> 4));
            v38 = (_QWORD *)(v36 + 8LL * v37);
            v39 = *v38;
            if ( *v35 != *v38 )
            {
              v58 = 1;
              v59 = 0;
              while ( v39 != -4096 )
              {
                if ( v39 == -8192 && !v59 )
                  v59 = v38;
                v37 = (a2 - 1) & (v58 + v37);
                v38 = (_QWORD *)(v36 + 8LL * v37);
                v39 = *v38;
                if ( *v35 == *v38 )
                  goto LABEL_54;
                ++v58;
              }
              if ( v59 )
                v38 = v59;
              v67[0] = (__int64)v38;
              v60 = *(_DWORD *)(v23 + 16);
              ++*(_QWORD *)v23;
              v40 = v60 + 1;
              if ( 4 * v40 < (unsigned int)(3 * a2) )
              {
                if ( (int)a2 - *(_DWORD *)(v23 + 20) - v40 <= (unsigned int)a2 >> 3 )
                {
LABEL_58:
                  sub_CE2A30(v23, a2);
                  a2 = (__int64)v35;
                  sub_DA5B20(v23, v35, v67);
                  v40 = *(_DWORD *)(v23 + 16) + 1;
                }
                *(_DWORD *)(v23 + 16) = v40;
                v41 = (__int64 *)v67[0];
                if ( *(_QWORD *)v67[0] != -4096 )
                  --*(_DWORD *)(v23 + 20);
                *v41 = *v35;
                goto LABEL_54;
              }
LABEL_57:
              LODWORD(a2) = 2 * a2;
              goto LABEL_58;
            }
LABEL_54:
            if ( v65 == ++v35 )
              goto LABEL_3;
          }
          v67[0] = 0;
          ++*(_QWORD *)v23;
          goto LABEL_57;
        }
      }
LABEL_3:
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v3 );
  }
}
