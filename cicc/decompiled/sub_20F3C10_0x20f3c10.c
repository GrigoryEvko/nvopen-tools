// Function: sub_20F3C10
// Address: 0x20f3c10
//
void __fastcall sub_20F3C10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  _QWORD *v4; // r9
  _DWORD *v6; // rdi
  __int64 *v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  _DWORD *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int16 v14; // ax
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdi
  unsigned int v19; // ecx
  unsigned int v20; // esi
  __int64 *v21; // rax
  __int64 v22; // r11
  __int64 v23; // r13
  __int64 *v24; // rdx
  __int64 v25; // r15
  int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdi
  __int64 (*v31)(); // rax
  __int64 v32; // rax
  __int32 v33; // edx
  int v34; // eax
  unsigned int v35; // edx
  __int64 v36; // r11
  int v37; // eax
  __int64 v38; // r10
  unsigned __int64 v39; // rcx
  __int64 v40; // r15
  unsigned int v41; // r15d
  __int64 v42; // rdx
  unsigned __int64 v43; // r13
  __int64 *v44; // rcx
  int v45; // r8d
  int v46; // r9d
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 *v49; // rax
  __int64 v50; // rax
  _QWORD *v51; // rsi
  _QWORD *v52; // rax
  __int64 v53; // rcx
  int v54; // r8d
  __int64 v55; // rax
  __int64 v56; // [rsp+0h] [rbp-110h]
  __int64 v57; // [rsp+8h] [rbp-108h]
  __int64 v58; // [rsp+8h] [rbp-108h]
  _QWORD *v59; // [rsp+10h] [rbp-100h]
  __int64 v60; // [rsp+18h] [rbp-F8h]
  __int64 v61; // [rsp+30h] [rbp-E0h]
  int v62; // [rsp+38h] [rbp-D8h]
  int v63; // [rsp+3Ch] [rbp-D4h]
  int v64; // [rsp+4Ch] [rbp-C4h] BYREF
  _QWORD *v65; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+58h] [rbp-B8h]
  _QWORD v67[22]; // [rsp+60h] [rbp-B0h] BYREF

  LODWORD(v3) = 1;
  v4 = v67;
  v66 = 0x800000001LL;
  v65 = v67;
  v67[0] = a2;
  v67[1] = a3;
  do
  {
    while ( 1 )
    {
      v6 = *(_DWORD **)(a1 + 120);
      v7 = &v4[2 * (unsigned int)v3 - 2];
      v8 = *v7;
      v9 = v7[1];
      LODWORD(v66) = v3 - 1;
      v61 = v9;
      v63 = *(_DWORD *)(v8 + 112);
      v64 = v63;
      v10 = &v6[*(unsigned int *)(a1 + 128)];
      if ( v10 == sub_20EA2B0(v6, (__int64)v10, &v64) )
        break;
      if ( !(_DWORD)v3 )
        goto LABEL_11;
    }
    sub_1DB9460(*(_QWORD *)(a1 + 104), v8, v61, **(_QWORD **)(*(_QWORD *)(a1 + 104) + 64LL), v3, v4);
    v11 = *(_QWORD *)(a1 + 64);
    if ( v63 < 0 )
      v12 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 16LL * (v63 & 0x7FFFFFFF) + 8);
    else
      v12 = *(_QWORD *)(*(_QWORD *)(v11 + 272) + 8LL * (unsigned int)v63);
    while ( v12 )
    {
      if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 && (*(_BYTE *)(v12 + 4) & 8) == 0 )
      {
        while ( 1 )
        {
          v25 = *(_QWORD *)(v12 + 16);
          do
            v12 = *(_QWORD *)(v12 + 32);
          while ( v12
               && ((*(_BYTE *)(v12 + 3) & 0x10) != 0 || (*(_BYTE *)(v12 + 4) & 8) != 0 || v25 == *(_QWORD *)(v12 + 16)) );
          v13 = *(_QWORD *)(v25 + 16);
          if ( *(_WORD *)v13 != 15 && (*(_WORD *)v13 != 1 || (*(_BYTE *)(*(_QWORD *)(v25 + 32) + 64LL) & 0x10) == 0) )
          {
            v14 = *(_WORD *)(v25 + 46);
            if ( (v14 & 4) != 0 || (v14 & 8) == 0 )
              v15 = (*(_QWORD *)(v13 + 8) >> 17) & 1LL;
            else
              LOBYTE(v15) = sub_1E15D00(v25, 0x20000u, 1);
            if ( !(_BYTE)v15 )
              goto LABEL_29;
          }
          v16 = v25;
          v17 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL);
          if ( (*(_BYTE *)(v25 + 46) & 4) != 0 )
          {
            do
              v16 = *(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(v16 + 46) & 4) != 0 );
          }
          v18 = *(_QWORD *)(v17 + 368);
          v19 = *(_DWORD *)(v17 + 384);
          if ( !v19 )
            goto LABEL_47;
          v20 = (v19 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v21 = (__int64 *)(v18 + 16LL * v20);
          v22 = *v21;
          if ( *v21 != v16 )
            break;
LABEL_26:
          v23 = v21[1];
          v24 = (__int64 *)sub_1DB3C70((__int64 *)v8, v23);
          if ( v24 == (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8))
            || (*(_DWORD *)((*v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v24 >> 1) & 3) > (*(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | (unsigned int)(v23 >> 1) & 3) )
          {
            if ( v61 )
              goto LABEL_29;
          }
          else if ( v61 != v24[2] )
          {
            goto LABEL_29;
          }
          v26 = sub_20EA270(v25, v63);
          v29 = (unsigned int)v26;
          if ( v26 )
          {
            if ( v26 >= 0 )
              goto LABEL_29;
            v35 = v26 & 0x7FFFFFFF;
            v36 = v26 & 0x7FFFFFFF;
            v37 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 312LL) + 4 * v36);
            if ( !v37 )
              v37 = v29;
            if ( *(_DWORD *)(a1 + 116) != v37 )
              goto LABEL_29;
            v38 = *(_QWORD *)(a1 + 16);
            v60 = 8LL * v35;
            v39 = *(unsigned int *)(v38 + 408);
            if ( v35 < (unsigned int)v39 )
            {
              v40 = *(_QWORD *)(*(_QWORD *)(v38 + 400) + 8LL * v35);
              if ( v40 )
                goto LABEL_57;
            }
            v41 = v35 + 1;
            if ( (unsigned int)v39 >= v35 + 1 )
              goto LABEL_55;
            v50 = v41;
            if ( v41 < v39 )
            {
              *(_DWORD *)(v38 + 408) = v41;
            }
            else if ( v41 > v39 )
            {
              if ( v41 > (unsigned __int64)*(unsigned int *)(v38 + 412) )
              {
                v56 = v36;
                v62 = v29;
                v58 = *(_QWORD *)(a1 + 16);
                sub_16CD150(v38 + 400, (const void *)(v38 + 416), v41, 8, v28, v29);
                v38 = v58;
                v36 = v56;
                LODWORD(v29) = v62;
                v50 = v41;
                v39 = *(unsigned int *)(v58 + 408);
              }
              v42 = *(_QWORD *)(v38 + 400);
              v51 = (_QWORD *)(v42 + 8 * v50);
              v52 = (_QWORD *)(v42 + 8 * v39);
              v53 = *(_QWORD *)(v38 + 416);
              if ( v51 != v52 )
              {
                do
                  *v52++ = v53;
                while ( v51 != v52 );
                v42 = *(_QWORD *)(v38 + 400);
              }
              *(_DWORD *)(v38 + 408) = v41;
LABEL_56:
              v57 = v36;
              v59 = (_QWORD *)v38;
              *(_QWORD *)(v42 + v60) = sub_1DBA290(v29);
              v40 = *(_QWORD *)(v59[50] + 8 * v57);
              sub_1DBB110(v59, v40);
LABEL_57:
              v43 = v23 & 0xFFFFFFFFFFFFFFF8LL;
              v44 = (__int64 *)sub_1DB3C70((__int64 *)v40, v43 | 4);
              if ( v44 == (__int64 *)(*(_QWORD *)v40 + 24LL * *(unsigned int *)(v40 + 8))
                || (*(_DWORD *)((*v44 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v44 >> 1) & 3) > (*(_DWORD *)(v43 + 24) | 2u) )
              {
                v47 = 0;
              }
              else
              {
                v47 = v44[2];
              }
              v48 = (unsigned int)v66;
              if ( (unsigned int)v66 >= HIDWORD(v66) )
              {
                sub_16CD150((__int64)&v65, v67, 0, 16, v45, v46);
                v48 = (unsigned int)v66;
              }
              v49 = &v65[2 * v48];
              *v49 = v40;
              v49[1] = v47;
              LODWORD(v66) = v66 + 1;
              goto LABEL_29;
            }
LABEL_55:
            v42 = *(_QWORD *)(v38 + 400);
            goto LABEL_56;
          }
          v30 = *(_QWORD *)(a1 + 72);
          v31 = *(__int64 (**)())(*(_QWORD *)v30 + 80LL);
          if ( v31 != sub_1EBAF80 )
            LODWORD(v29) = ((__int64 (__fastcall *)(__int64, __int64, int *, __int64, __int64, __int64))v31)(
                             v30,
                             v25,
                             &v64,
                             v27,
                             v28,
                             v29);
          if ( (_DWORD)v29 == v63 && *(_DWORD *)(a1 + 112) == v64 )
          {
            *(_QWORD *)(v25 + 16) = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL) + 384LL;
            v32 = *(unsigned int *)(a1 + 384);
            if ( (unsigned int)v32 >= *(_DWORD *)(a1 + 388) )
            {
              sub_16CD150(a1 + 376, (const void *)(a1 + 392), 0, 8, v28, v29);
              v32 = *(unsigned int *)(a1 + 384);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 376) + 8 * v32) = v25;
            v33 = *(_DWORD *)(a1 + 112);
            ++*(_DWORD *)(a1 + 384);
            sub_20F2770(a1 + 456, v25, v33);
          }
LABEL_29:
          if ( !v12 )
            goto LABEL_10;
        }
        v34 = 1;
        while ( v22 != -8 )
        {
          v54 = v34 + 1;
          v55 = (v19 - 1) & (v20 + v34);
          v20 = v55;
          v21 = (__int64 *)(v18 + 16 * v55);
          v22 = *v21;
          if ( *v21 == v16 )
            goto LABEL_26;
          v34 = v54;
        }
LABEL_47:
        v21 = (__int64 *)(v18 + 16LL * v19);
        goto LABEL_26;
      }
      v12 = *(_QWORD *)(v12 + 32);
    }
LABEL_10:
    LODWORD(v3) = v66;
    v4 = v65;
  }
  while ( (_DWORD)v66 );
LABEL_11:
  if ( v4 != v67 )
    _libc_free((unsigned __int64)v4);
}
