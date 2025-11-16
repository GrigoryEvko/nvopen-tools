// Function: sub_A1FDA0
// Address: 0xa1fda0
//
__int64 __fastcall sub_A1FDA0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rsi
  int v6; // edi
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rcx
  int v11; // r11d
  unsigned int i; // eax
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  int v28; // r14d
  __int64 v29; // r15
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // rdx
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // rdx
  bool v37; // cf
  __int64 v38; // r8
  __int64 v39; // rdx
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rdx
  _BYTE *v43; // rcx
  __int64 j; // rax
  __int64 v45; // r14
  __int64 v46; // rax
  _BYTE *v47; // rcx
  __int64 k; // rax
  int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rsi
  int v53; // eax
  __int64 v54; // rax
  __int64 v55; // rsi
  int v56; // eax
  int v57; // eax
  __int64 v58; // rsi
  __int64 v59; // r14
  __int64 v60; // rdx
  __int64 v61; // r15
  __int64 v62; // r15
  __int64 v63; // rsi
  __int64 v64; // [rsp+0h] [rbp-2A0h]
  __int64 v65; // [rsp+8h] [rbp-298h]
  __int64 v66; // [rsp+10h] [rbp-290h]
  __int64 v67; // [rsp+18h] [rbp-288h]
  __int64 v68; // [rsp+18h] [rbp-288h]
  __int64 v69; // [rsp+18h] [rbp-288h]
  __int64 v70; // [rsp+18h] [rbp-288h]
  unsigned int *v71; // [rsp+28h] [rbp-278h]
  unsigned int *v73; // [rsp+40h] [rbp-260h]
  __int64 v74; // [rsp+48h] [rbp-258h]
  __int64 v75; // [rsp+50h] [rbp-250h] BYREF
  __int64 v76; // [rsp+58h] [rbp-248h] BYREF
  _BYTE *v77; // [rsp+60h] [rbp-240h] BYREF
  __int64 v78; // [rsp+68h] [rbp-238h]
  _BYTE v79[560]; // [rsp+70h] [rbp-230h] BYREF

  result = *(_QWORD *)(a1 + 392);
  if ( *(_QWORD *)(a1 + 384) != result )
  {
    sub_A19830(*(_QWORD *)a1, 0xAu, 3u);
    v5 = *(_QWORD *)(a1 + 392);
    v77 = v79;
    v78 = 0x4000000000LL;
    v71 = (unsigned int *)v5;
    if ( *(_QWORD *)(a1 + 384) != v5 )
    {
      v73 = *(unsigned int **)(a1 + 384);
      v5 = a1 + 24;
      v6 = 64;
      v65 = a1 + 24;
      while ( 1 )
      {
        v7 = 0;
        v8 = *((_QWORD *)v73 + 1);
        v9 = *v73;
        v75 = v8;
        if ( v8 )
        {
          v10 = *(unsigned int *)(a1 + 376);
          v3 = *(_QWORD *)(a1 + 360);
          if ( (_DWORD)v10 )
          {
            v11 = 1;
            for ( i = (v10 - 1)
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned __int64)(unsigned int)(37 * v9) << 32)
                        | ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))) >> 31)
                     ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))); ; i = (v10 - 1) & v13 )
            {
              v5 = v3 + 24LL * i;
              v4 = *(unsigned int *)v5;
              if ( (_DWORD)v9 == (_DWORD)v4 && v8 == *(_QWORD *)(v5 + 8) )
                break;
              if ( (_DWORD)v4 == -1 && *(_QWORD *)(v5 + 8) == -4 )
                goto LABEL_11;
              v13 = v11 + i;
              ++v11;
            }
          }
          else
          {
LABEL_11:
            v5 = v3 + 24 * v10;
          }
          v7 = *(unsigned int *)(v5 + 16);
        }
        v14 = 0;
        if ( !v6 )
        {
          v5 = (__int64)v79;
          sub_C8D5F0(&v77, v79, 1, 8);
          v14 = (unsigned int)v78;
        }
        *(_QWORD *)&v77[8 * v14] = v7;
        v15 = HIDWORD(v78);
        LODWORD(v78) = v78 + 1;
        v16 = (unsigned int)v78;
        if ( (unsigned __int64)(unsigned int)v78 + 1 > HIDWORD(v78) )
        {
          v5 = (__int64)v79;
          sub_C8D5F0(&v77, v79, (unsigned int)v78 + 1LL, 8);
          v16 = (unsigned int)v78;
        }
        v17 = (__int64)v77;
        *(_QWORD *)&v77[8 * v16] = v9;
        LODWORD(v78) = v78 + 1;
        v18 = (__int64 *)sub_A73280(&v75, v5, v17, v15, v3, v4, v64, v65, v66);
        v74 = sub_A73290(&v75);
        if ( v18 != (__int64 *)v74 )
          break;
LABEL_54:
        v5 = 3;
        sub_A1FB70(*(_QWORD *)a1, 3u, (__int64)&v77, 0);
        v73 += 4;
        LODWORD(v78) = 0;
        if ( v71 == v73 )
          goto LABEL_65;
        v6 = HIDWORD(v78);
      }
      while ( 1 )
      {
        v76 = *v18;
        if ( (unsigned __int8)sub_A71800(&v76) )
          break;
        if ( (unsigned __int8)sub_A71820(&v76) )
        {
          v26 = (unsigned int)v78;
          v27 = (unsigned int)v78 + 1LL;
          if ( v27 > HIDWORD(v78) )
          {
            sub_C8D5F0(&v77, v79, v27, 8);
            v26 = (unsigned int)v78;
          }
          *(_QWORD *)&v77[8 * v26] = 1;
          LODWORD(v78) = v78 + 1;
          v28 = sub_A71AE0(&v76);
          v29 = sub_A15330(v28);
          v30 = (unsigned int)v78;
          v31 = (unsigned int)v78 + 1LL;
          if ( v31 > HIDWORD(v78) )
          {
            sub_C8D5F0(&v77, v79, v31, 8);
            v30 = (unsigned int)v78;
          }
          *(_QWORD *)&v77[8 * v30] = v29;
          LODWORD(v78) = v78 + 1;
          if ( v28 != 92 )
          {
            v22 = sub_A71B80(&v76);
            goto LABEL_22;
          }
          v23 = sub_A71B80(&v76) | 0x100000000000000LL;
LABEL_23:
          v24 = (unsigned int)v78;
          v25 = (unsigned int)v78 + 1LL;
          if ( v25 > HIDWORD(v78) )
          {
            sub_C8D5F0(&v77, v79, v25, 8);
            v24 = (unsigned int)v78;
          }
          *(_QWORD *)&v77[8 * v24] = v23;
          LODWORD(v78) = v78 + 1;
          goto LABEL_26;
        }
        if ( !(unsigned __int8)sub_A71840(&v76) )
        {
          if ( (unsigned __int8)sub_A71860(&v76) )
          {
            if ( sub_A72A60(&v76) )
            {
              sub_A188E0((__int64)&v77, 6);
              v49 = sub_A71AE0(&v76);
              v50 = sub_A15330(v49);
              sub_A188E0((__int64)&v77, v50);
              v51 = sub_A72A60(&v76);
              v52 = (unsigned int)sub_A172F0(v65, v51);
            }
            else
            {
              sub_A188E0((__int64)&v77, 5);
              v56 = sub_A71AE0(&v76);
              v52 = sub_A15330(v56);
            }
            sub_A188E0((__int64)&v77, v52);
          }
          else if ( (unsigned __int8)sub_A71880(&v76) )
          {
            sub_A188E0((__int64)&v77, 7);
            v53 = sub_A71AE0(&v76);
            v54 = sub_A15330(v53);
            sub_A188E0((__int64)&v77, v54);
            v55 = sub_A72A90(&v76);
            sub_A18930((__int64 *)&v77, v55, 1);
          }
          else
          {
            sub_A188E0((__int64)&v77, 8);
            v57 = sub_A71AE0(&v76);
            v58 = sub_A15330(v57);
            sub_A188E0((__int64)&v77, v58);
            v59 = sub_A72AC0(&v76);
            v61 = v60;
            sub_A188E0((__int64)&v77, v60);
            v62 = v59 + 32 * v61;
            sub_A188E0((__int64)&v77, *(unsigned int *)(v59 + 8));
            while ( v62 != v59 )
            {
              v63 = v59;
              v59 += 32;
              sub_A18930((__int64 *)&v77, v63, 0);
            }
          }
          goto LABEL_26;
        }
        v32 = sub_A71FD0(&v76);
        v34 = v33;
        v35 = sub_A72240(&v76);
        v37 = v36 == 0;
        v38 = v36;
        v39 = (unsigned int)v78;
        v40 = v35;
        v41 = 3 - (v37 - 1LL);
        if ( (unsigned __int64)(unsigned int)v78 + 1 > HIDWORD(v78) )
        {
          v64 = v38;
          v66 = v40;
          v70 = 3 - (v37 - 1LL);
          sub_C8D5F0(&v77, v79, (unsigned int)v78 + 1LL, 8);
          v39 = (unsigned int)v78;
          v38 = v64;
          v40 = v66;
          v41 = v70;
        }
        *(_QWORD *)&v77[8 * v39] = v41;
        LODWORD(v78) = v78 + 1;
        v42 = (unsigned int)v78;
        if ( v34 + (unsigned __int64)(unsigned int)v78 > HIDWORD(v78) )
        {
          v66 = v38;
          v67 = v40;
          sub_C8D5F0(&v77, v79, v34 + (unsigned int)v78, 8);
          v42 = (unsigned int)v78;
          v38 = v66;
          v40 = v67;
        }
        v43 = &v77[8 * v42];
        if ( v34 > 0 )
        {
          for ( j = 0; j != v34; ++j )
            *(_QWORD *)&v43[8 * j] = *(char *)(v32 + j);
          LODWORD(v42) = v78;
        }
        LODWORD(v78) = v42 + v34;
        v45 = (unsigned int)(v42 + v34);
        if ( v45 + 1 > (unsigned __int64)HIDWORD(v78) )
        {
          v66 = v38;
          v69 = v40;
          sub_C8D5F0(&v77, v79, v45 + 1, 8);
          v45 = (unsigned int)v78;
          v38 = v66;
          v40 = v69;
        }
        *(_QWORD *)&v77[8 * v45] = 0;
        v46 = (unsigned int)(v78 + 1);
        LODWORD(v78) = v78 + 1;
        if ( v38 )
        {
          if ( v38 + v46 > (unsigned __int64)HIDWORD(v78) )
          {
            v66 = v38;
            v68 = v40;
            sub_C8D5F0(&v77, v79, v38 + v46, 8);
            v46 = (unsigned int)v78;
            v38 = v66;
            v40 = v68;
          }
          v47 = &v77[8 * v46];
          if ( v38 > 0 )
          {
            for ( k = 0; k != v38; ++k )
              *(_QWORD *)&v47[8 * k] = *(char *)(v40 + k);
            LODWORD(v46) = v78;
          }
          ++v18;
          LODWORD(v78) = v46 + v38;
          sub_A188E0((__int64)&v77, 0);
          if ( (__int64 *)v74 == v18 )
            goto LABEL_54;
        }
        else
        {
LABEL_26:
          if ( (__int64 *)v74 == ++v18 )
            goto LABEL_54;
        }
      }
      v19 = (unsigned int)v78;
      v20 = (unsigned int)v78 + 1LL;
      if ( v20 > HIDWORD(v78) )
      {
        sub_C8D5F0(&v77, v79, v20, 8);
        v19 = (unsigned int)v78;
      }
      *(_QWORD *)&v77[8 * v19] = 0;
      LODWORD(v78) = v78 + 1;
      v21 = sub_A71AE0(&v76);
      v22 = sub_A15330(v21);
LABEL_22:
      v23 = v22;
      goto LABEL_23;
    }
LABEL_65:
    result = sub_A192A0(*(_QWORD *)a1);
    if ( v77 != v79 )
      return _libc_free(v77, v5);
  }
  return result;
}
