// Function: sub_1490E00
// Address: 0x1490e00
//
__int64 __fastcall sub_1490E00(_QWORD *a1, __m128i a2, __m128i a3)
{
  __int64 v4; // rdx
  _BYTE *v5; // rdi
  _BYTE *v6; // rax
  _BYTE *v7; // r13
  int v8; // edx
  size_t v9; // r15
  __int64 v10; // r12
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r15
  _BYTE *v16; // r9
  _BYTE *v17; // r13
  size_t v18; // r10
  unsigned __int64 v19; // r12
  unsigned int v20; // ecx
  __int64 *v21; // rdx
  __int64 v22; // r10
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rax
  unsigned __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r12
  int v46; // edx
  __int64 v47; // rax
  unsigned __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // rax
  int v51; // eax
  int v52; // r11d
  __int64 v53; // [rsp+0h] [rbp-530h]
  size_t v54; // [rsp+8h] [rbp-528h]
  __int64 v55; // [rsp+8h] [rbp-528h]
  int v56; // [rsp+10h] [rbp-520h]
  _BYTE *v57; // [rsp+10h] [rbp-520h]
  __int64 v58; // [rsp+28h] [rbp-508h]
  _BYTE *v59; // [rsp+28h] [rbp-508h]
  __int64 v60; // [rsp+30h] [rbp-500h] BYREF
  __int64 v61; // [rsp+38h] [rbp-4F8h] BYREF
  _QWORD *v62; // [rsp+40h] [rbp-4F0h] BYREF
  __int64 v63; // [rsp+48h] [rbp-4E8h] BYREF
  __int64 v64; // [rsp+50h] [rbp-4E0h]
  __int64 v65; // [rsp+58h] [rbp-4D8h]
  unsigned int v66; // [rsp+60h] [rbp-4D0h]
  _BYTE v67[16]; // [rsp+70h] [rbp-4C0h] BYREF
  __int64 v68; // [rsp+80h] [rbp-4B0h]
  _BYTE *v69; // [rsp+A0h] [rbp-490h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-488h]
  _BYTE dest[64]; // [rsp+B0h] [rbp-480h] BYREF
  _QWORD v72[136]; // [rsp+F0h] [rbp-440h] BYREF

  sub_1457DF0((__int64)v72, a1[3], a1[5], a1[6], a1[7], a1[8]);
  v4 = a1[8];
  v5 = dest;
  v6 = *(_BYTE **)(v4 + 40);
  v7 = *(_BYTE **)(v4 + 32);
  v8 = 0;
  v69 = dest;
  v70 = 0x800000000LL;
  v9 = v6 - v7;
  v10 = (v6 - v7) >> 3;
  if ( (unsigned __int64)(v6 - v7) > 0x40 )
  {
    v59 = v6;
    sub_16CD150(&v69, dest, (v6 - v7) >> 3, 8);
    v8 = v70;
    v6 = v59;
    v5 = &v69[8 * (unsigned int)v70];
  }
  if ( v7 != v6 )
  {
    memmove(v5, v7, v9);
    v8 = v70;
  }
  v62 = v72;
  v11 = v10 + v8;
  LODWORD(v70) = v10 + v8;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  if ( (_DWORD)v10 + v8 )
  {
    while ( 1 )
    {
      v12 = (unsigned __int64)v69;
      v13 = v11;
      v14 = v11 - 1;
      v15 = *(_QWORD *)&v69[8 * v13 - 8];
      LODWORD(v70) = v14;
      v16 = *(_BYTE **)(v15 + 16);
      v17 = *(_BYTE **)(v15 + 8);
      v18 = v16 - v17;
      v19 = (v16 - v17) >> 3;
      if ( v19 > (unsigned __int64)HIDWORD(v70) - v14 )
      {
        v54 = *(_QWORD *)(v15 + 16) - (_QWORD)v17;
        v57 = *(_BYTE **)(v15 + 16);
        sub_16CD150(&v69, dest, v14 + v19, 8);
        v12 = (unsigned __int64)v69;
        v14 = (unsigned int)v70;
        v18 = v54;
        v16 = v57;
      }
      if ( v17 != v16 )
      {
        memmove((void *)(v12 + 8 * v14), v17, v18);
        LODWORD(v14) = v70;
      }
      LODWORD(v70) = v19 + v14;
      v60 = sub_1481F60(a1, v15, a2, a3);
      if ( !v66 )
        goto LABEL_31;
      v20 = (v66 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
      v21 = (__int64 *)(v64 + 16LL * v20);
      v22 = *v21;
      if ( v60 != *v21 )
        break;
LABEL_12:
      if ( v21 == (__int64 *)(v64 + 16LL * v66) )
        goto LABEL_31;
      v23 = v21[1];
LABEL_14:
      v24 = sub_1481F60(v72, v15, a2, a3);
      if ( v23 != sub_1456E90((__int64)v72) && v24 != sub_1456E90((__int64)v72) )
      {
        v67[0] = 0;
        sub_145AC20(v23, v67);
        if ( !v67[0] )
        {
          sub_145AC20(v24, v67);
          if ( !v67[0] )
          {
            v25 = sub_1456040(v23);
            v26 = sub_1456C90((__int64)a1, v25);
            v27 = sub_1456040(v24);
            if ( v26 <= sub_1456C90((__int64)a1, v27) )
            {
              v47 = sub_1456040(v23);
              v48 = sub_1456C90((__int64)a1, v47);
              v49 = sub_1456040(v24);
              if ( v48 < sub_1456C90((__int64)a1, v49) )
              {
                v50 = sub_1456040(v24);
                v23 = sub_14747F0((__int64)v72, v23, v50, 0);
              }
            }
            else
            {
              v28 = sub_1456040(v23);
              v24 = sub_14747F0((__int64)v72, v24, v28, 0);
            }
            v29 = v72;
            v30 = sub_14806B0((__int64)v72, v23, v24, 0, 0);
            v31 = v30;
            if ( !*(_WORD *)(v30 + 24) )
            {
              v32 = *(_QWORD *)(v30 + 32);
              v33 = *(unsigned int *)(v32 + 32);
              v56 = *(_DWORD *)(v32 + 32);
              if ( (unsigned int)v33 > 0x40 )
              {
                v29 = (_QWORD *)(v32 + 24);
                v55 = *(_QWORD *)(v30 + 32);
                v53 = v30;
                v51 = sub_16A57B0(v32 + 24);
                v31 = v53;
                v33 = (unsigned int)(v56 - v51);
                if ( (unsigned int)v33 > 0x40 )
                {
LABEL_24:
                  v58 = v31;
                  v35 = sub_16BA580(v29, v23, v33);
                  sub_1263B40(v35, "Trip Count Changed!\n");
                  v37 = sub_16BA580(v35, "Trip Count Changed!\n", v36);
                  v38 = sub_1263B40(v37, "Old: ");
                  sub_1456620(v23, v38);
                  sub_1263B40(v38, "\n");
                  v40 = sub_16BA580(v38, "\n", v39);
                  v41 = sub_1263B40(v40, "New: ");
                  sub_1456620(v24, v41);
                  sub_1263B40(v41, "\n");
                  v43 = sub_16BA580(v41, "\n", v42);
                  v44 = sub_1263B40(v43, "Delta: ");
                  sub_1456620(v58, v44);
                  sub_1263B40(v44, "\n");
                  abort();
                }
                v34 = **(_QWORD **)(v55 + 24);
              }
              else
              {
                v34 = *(_QWORD *)(v32 + 24);
              }
              if ( v34 )
                goto LABEL_24;
            }
          }
        }
      }
      v11 = v70;
      if ( !(_DWORD)v70 )
        goto LABEL_26;
    }
    v46 = 1;
    while ( v22 != -8 )
    {
      v52 = v46 + 1;
      v20 = (v66 - 1) & (v46 + v20);
      v21 = (__int64 *)(v64 + 16LL * v20);
      v22 = *v21;
      if ( v60 == *v21 )
        goto LABEL_12;
      v46 = v52;
    }
LABEL_31:
    v61 = sub_1490810((__int64 *)&v62, v60, a2, a3);
    sub_1466830((__int64)v67, (__int64)&v63, &v60, &v61);
    v23 = *(_QWORD *)(v68 + 8);
    goto LABEL_14;
  }
LABEL_26:
  j___libc_free_0(v64);
  if ( v69 != dest )
    _libc_free((unsigned __int64)v69);
  return sub_14602B0((__int64)v72);
}
