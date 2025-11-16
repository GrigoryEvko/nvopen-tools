// Function: sub_34FAE40
// Address: 0x34fae40
//
void __fastcall sub_34FAE40(__int64 a1, __int64 a2, __int64 a3)
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
  unsigned __int64 j; // rdx
  unsigned __int64 v14; // r15
  int i; // esi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 (__fastcall *v18)(__int64); // rax
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned __int64 k; // rax
  __int64 m; // rdi
  __int16 v24; // dx
  __int64 v25; // rdi
  unsigned int v26; // esi
  unsigned int v27; // ecx
  __int64 *v28; // rdx
  __int64 v29; // r9
  __int64 v30; // r13
  __int64 *v31; // rdx
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // r8
  int v35; // r11d
  __int64 v36; // rdi
  __int64 (*v37)(); // rax
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int32 v41; // edx
  int v42; // edx
  unsigned int v43; // eax
  int v44; // edx
  __int64 v45; // r9
  unsigned __int64 v46; // rcx
  unsigned int v47; // eax
  __int64 v48; // rdx
  __int64 v49; // r15
  unsigned __int64 v50; // r13
  __int64 *v51; // rdx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // r13
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 *v57; // rax
  int v58; // r8d
  __int64 v59; // r10
  unsigned __int64 v60; // r15
  _QWORD *v61; // rsi
  _QWORD *v62; // rdi
  __int64 v63; // [rsp+8h] [rbp-118h]
  _QWORD *v64; // [rsp+10h] [rbp-110h]
  __int64 v65; // [rsp+10h] [rbp-110h]
  __int64 v66; // [rsp+18h] [rbp-108h]
  __int64 *v67; // [rsp+18h] [rbp-108h]
  int v68; // [rsp+30h] [rbp-F0h]
  int v69; // [rsp+34h] [rbp-ECh]
  __int64 v70; // [rsp+38h] [rbp-E8h]
  int v71[4]; // [rsp+40h] [rbp-E0h] BYREF
  char v72; // [rsp+50h] [rbp-D0h]
  _QWORD *v73; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v74; // [rsp+68h] [rbp-B8h]
  _QWORD v75[22]; // [rsp+70h] [rbp-B0h] BYREF

  LODWORD(v3) = 1;
  v4 = v75;
  v74 = 0x800000001LL;
  v73 = v75;
  v75[0] = a2;
  v75[1] = a3;
  do
  {
    while ( 1 )
    {
      v6 = *(_DWORD **)(a1 + 88);
      v7 = &v4[2 * (unsigned int)v3 - 2];
      v8 = *v7;
      v9 = v7[1];
      LODWORD(v74) = v3 - 1;
      v70 = v9;
      v69 = *(_DWORD *)(v8 + 112);
      v71[0] = v69;
      v10 = &v6[*(unsigned int *)(a1 + 96)];
      if ( v10 == sub_34F4EB0(v6, (__int64)v10, v71) )
        break;
      if ( !(_DWORD)v3 )
        goto LABEL_11;
    }
    sub_2E0FCD0(*(_QWORD *)(a1 + 72), v8, v70, **(_QWORD **)(*(_QWORD *)(a1 + 72) + 64LL), v3, v4);
    v11 = *(_QWORD *)(a1 + 40);
    if ( v69 < 0 )
      v12 = *(_QWORD *)(*(_QWORD *)(v11 + 56) + 16LL * (v69 & 0x7FFFFFFF) + 8);
    else
      v12 = *(_QWORD *)(*(_QWORD *)(v11 + 304) + 8LL * (unsigned int)v69);
    while ( v12 )
    {
      if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 && (*(_BYTE *)(v12 + 4) & 8) == 0 )
      {
        while ( 1 )
        {
          v14 = *(_QWORD *)(v12 + 16);
          v19 = v14;
          for ( i = *(_DWORD *)(v14 + 44) & 0xFFFFFF;
                (*(_BYTE *)(v19 + 44) & 4) != 0;
                v19 = *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL )
          {
            ;
          }
          while ( 1 )
          {
            v12 = *(_QWORD *)(v12 + 32);
            if ( !v12 )
              break;
            if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 && (*(_BYTE *)(v12 + 4) & 8) == 0 )
            {
              for ( j = *(_QWORD *)(v12 + 16); (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
                ;
              if ( j != v19 )
                break;
            }
          }
          for ( ; (*(_DWORD *)(v14 + 44) & 4) != 0; i = *(_DWORD *)(v14 + 44) & 0xFFFFFF )
            v14 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (unsigned int)*(unsigned __int16 *)(v14 + 68) - 1 > 1
            || (*(_BYTE *)(*(_QWORD *)(v14 + 32) + 64LL) & 0x10) == 0 )
          {
            if ( (i & 8) != 0 )
              LOBYTE(v16) = sub_2E88A90(v14, 0x100000, 1);
            else
              v16 = (*(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL) >> 20) & 1LL;
            if ( !(_BYTE)v16 && *(_WORD *)(v14 + 68) != 20 )
            {
              v17 = *(_QWORD *)(a1 + 48);
              v18 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 520LL);
              if ( v18 == sub_2DCA430 )
                goto LABEL_29;
              ((void (__fastcall *)(int *, __int64, unsigned __int64))v18)(v71, v17, v14);
              if ( !v72 )
                goto LABEL_29;
            }
          }
          v20 = v14;
          v21 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
          for ( k = v14; (*(_BYTE *)(k + 44) & 4) != 0; k = *(_QWORD *)k & 0xFFFFFFFFFFFFFFF8LL )
            ;
          if ( (*(_DWORD *)(v14 + 44) & 8) != 0 )
          {
            do
              v20 = *(_QWORD *)(v20 + 8);
            while ( (*(_BYTE *)(v20 + 44) & 8) != 0 );
          }
          for ( m = *(_QWORD *)(v20 + 8); m != k; k = *(_QWORD *)(k + 8) )
          {
            v24 = *(_WORD *)(k + 68);
            if ( (unsigned __int16)(v24 - 14) > 4u && v24 != 24 )
              break;
          }
          v25 = *(_QWORD *)(v21 + 128);
          v26 = *(_DWORD *)(v21 + 144);
          if ( !v26 )
            goto LABEL_60;
          v27 = (v26 - 1) & (((unsigned int)k >> 9) ^ ((unsigned int)k >> 4));
          v28 = (__int64 *)(v25 + 16LL * v27);
          v29 = *v28;
          if ( *v28 != k )
            break;
LABEL_44:
          v30 = v28[1];
          v31 = (__int64 *)sub_2E09D00((__int64 *)v8, v30);
          if ( v31 == (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8))
            || (*(_DWORD *)((*v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v31 >> 1) & 3) > (*(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | (unsigned int)(v30 >> 1) & 3) )
          {
            v32 = 0;
          }
          else
          {
            v32 = v31[2];
          }
          if ( v70 == v32 )
          {
            v33 = sub_34F55E0(v14, v69, *(_QWORD *)(a1 + 48));
            v35 = v33;
            if ( v33 )
            {
              if ( v33 >= 0 )
                goto LABEL_29;
              v43 = v33 & 0x7FFFFFFF;
              v44 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 80LL) + 4LL * v43);
              if ( !v44 )
                v44 = v35;
              if ( v44 != *(_DWORD *)(a1 + 84) )
                goto LABEL_29;
              v45 = *(_QWORD *)(a1 + 16);
              v66 = 8LL * v43;
              v46 = *(unsigned int *)(v45 + 160);
              if ( v43 >= (unsigned int)v46 || (v49 = *(_QWORD *)(*(_QWORD *)(v45 + 152) + 8LL * v43)) == 0 )
              {
                v47 = v43 + 1;
                if ( (unsigned int)v46 >= v47 || v47 == v46 )
                {
LABEL_67:
                  v48 = *(_QWORD *)(v45 + 152);
                }
                else
                {
                  if ( v47 < v46 )
                  {
                    *(_DWORD *)(v45 + 160) = v47;
                    goto LABEL_67;
                  }
                  v59 = *(_QWORD *)(v45 + 168);
                  v60 = v47 - v46;
                  if ( v47 > (unsigned __int64)*(unsigned int *)(v45 + 164) )
                  {
                    v63 = *(_QWORD *)(v45 + 168);
                    v68 = v35;
                    v65 = *(_QWORD *)(a1 + 16);
                    sub_C8D5F0(v45 + 152, (const void *)(v45 + 168), v47, 8u, v34, v45);
                    v45 = v65;
                    v59 = v63;
                    v35 = v68;
                    v46 = *(unsigned int *)(v65 + 160);
                  }
                  v48 = *(_QWORD *)(v45 + 152);
                  v61 = (_QWORD *)(v48 + 8 * v46);
                  v62 = &v61[v60];
                  if ( v61 != v62 )
                  {
                    do
                      *v61++ = v59;
                    while ( v62 != v61 );
                    LODWORD(v46) = *(_DWORD *)(v45 + 160);
                    v48 = *(_QWORD *)(v45 + 152);
                  }
                  *(_DWORD *)(v45 + 160) = v46 + v60;
                }
                v64 = (_QWORD *)v45;
                v67 = (__int64 *)(v66 + v48);
                v49 = sub_2E10F30(v35);
                *v67 = v49;
                sub_2E11E80(v64, v49);
              }
              v50 = v30 & 0xFFFFFFFFFFFFFFF8LL;
              v51 = (__int64 *)sub_2E09D00((__int64 *)v49, v50 | 4);
              if ( v51 == (__int64 *)(*(_QWORD *)v49 + 24LL * *(unsigned int *)(v49 + 8))
                || (*(_DWORD *)((*v51 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v51 >> 1) & 3) > (*(_DWORD *)(v50 + 24) | 2u) )
              {
                v54 = 0;
              }
              else
              {
                v54 = v51[2];
              }
              v55 = (unsigned int)v74;
              v56 = (unsigned int)v74 + 1LL;
              if ( v56 > HIDWORD(v74) )
              {
                sub_C8D5F0((__int64)&v73, v75, v56, 0x10u, v52, v53);
                v55 = (unsigned int)v74;
              }
              v57 = &v73[2 * v55];
              *v57 = v49;
              v57[1] = v54;
              LODWORD(v74) = v74 + 1;
              goto LABEL_29;
            }
            v36 = *(_QWORD *)(a1 + 48);
            v37 = *(__int64 (**)())(*(_QWORD *)v36 + 120LL);
            if ( v37 != sub_2F4C0B0 )
              v35 = ((__int64 (__fastcall *)(__int64, unsigned __int64, int *))v37)(v36, v14, v71);
            if ( v69 == v35 && *(_DWORD *)(a1 + 80) == v71[0] )
            {
              sub_2E88D70(v14, (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL) - 280LL));
              v40 = *(unsigned int *)(a1 + 384);
              if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 388) )
              {
                sub_C8D5F0(a1 + 376, (const void *)(a1 + 392), v40 + 1, 8u, v38, v39);
                v40 = *(unsigned int *)(a1 + 384);
              }
              *(_QWORD *)(*(_QWORD *)(a1 + 376) + 8 * v40) = v14;
              v41 = *(_DWORD *)(a1 + 80);
              ++*(_DWORD *)(a1 + 384);
              sub_34FA050(a1 + 456, v14, v41);
            }
          }
LABEL_29:
          if ( !v12 )
            goto LABEL_10;
        }
        v42 = 1;
        while ( v29 != -4096 )
        {
          v58 = v42 + 1;
          v27 = (v26 - 1) & (v27 + v42);
          v28 = (__int64 *)(v25 + 16LL * v27);
          v29 = *v28;
          if ( *v28 == k )
            goto LABEL_44;
          v42 = v58;
        }
LABEL_60:
        v28 = (__int64 *)(v25 + 16LL * v26);
        goto LABEL_44;
      }
      v12 = *(_QWORD *)(v12 + 32);
    }
LABEL_10:
    LODWORD(v3) = v74;
    v4 = v73;
  }
  while ( (_DWORD)v74 );
LABEL_11:
  if ( v4 != v75 )
    _libc_free((unsigned __int64)v4);
}
