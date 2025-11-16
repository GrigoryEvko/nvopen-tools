// Function: sub_2E57DC0
// Address: 0x2e57dc0
//
void __fastcall sub_2E57DC0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // r11
  __int64 *v15; // rbx
  __int64 *v16; // r8
  __int64 v17; // rcx
  __int64 *v18; // r9
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 *v21; // r12
  int v22; // edi
  unsigned int v23; // esi
  __int64 *v24; // rdx
  __int64 v25; // r11
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  int v29; // edi
  __int64 v30; // r8
  unsigned int v31; // esi
  int v32; // eax
  unsigned int v33; // r10d
  int *v34; // rdx
  int v35; // edi
  __int64 v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  int v47; // edx
  int v48; // r10d
  int v49; // edi
  int v50; // edi
  int v51; // [rsp+10h] [rbp-C0h]
  __int64 v52; // [rsp+10h] [rbp-C0h]
  __int64 v53; // [rsp+20h] [rbp-B0h]
  __int64 v54; // [rsp+20h] [rbp-B0h]
  __int64 v55; // [rsp+28h] [rbp-A8h]
  __int64 *v56; // [rsp+28h] [rbp-A8h]
  int v57; // [rsp+3Ch] [rbp-94h] BYREF
  __int64 v58; // [rsp+40h] [rbp-90h] BYREF
  __int64 v59; // [rsp+48h] [rbp-88h] BYREF
  unsigned __int64 v60[2]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE v61[48]; // [rsp+60h] [rbp-70h] BYREF
  int v62; // [rsp+90h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 16);
  v5 = **(_QWORD **)(a2 + 32);
  v58 = v5;
  if ( v5 )
  {
    v6 = (unsigned int)(*(_DWORD *)(v5 + 24) + 1);
    v7 = *(_DWORD *)(v5 + 24) + 1;
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  if ( v7 < *(_DWORD *)(v4 + 32) && *(_QWORD *)(*(_QWORD *)(v4 + 24) + 8 * v6) )
  {
    v8 = *sub_2E57C80(a1 + 112, &v58);
    v60[0] = (unsigned __int64)v61;
    v11 = 0x600000000LL;
    v12 = *(unsigned int *)(v8 + 32);
    v55 = v8;
    v60[1] = 0x600000000LL;
    if ( (_DWORD)v12 )
      sub_2E4EFA0((__int64)v60, v8 + 24, v12, 0x600000000LL, v9, v10);
    v62 = *(_DWORD *)(v55 + 88);
    v13 = *(_QWORD *)(v58 + 56);
    v14 = v58 + 48;
    v53 = a1 + 56;
    if ( v13 != v58 + 48 )
    {
      while ( 1 )
      {
        if ( *(_WORD *)(v13 + 68) != 68 && *(_WORD *)(v13 + 68) )
          goto LABEL_10;
        v31 = *(_DWORD *)(a1 + 80);
        v32 = *(_DWORD *)(*(_QWORD *)(v13 + 32) + 8LL);
        v57 = v32;
        if ( !v31 )
          break;
        v9 = *(_QWORD *)(a1 + 64);
        v10 = v31 - 1;
        v33 = v10 & (37 * v32);
        v34 = (int *)(v9 + 8LL * v33);
        v35 = *v34;
        if ( v32 != *v34 )
        {
          v51 = 1;
          v11 = 0;
          while ( v35 != -1 )
          {
            if ( !v11 && v35 == -2 )
              v11 = (__int64)v34;
            v33 = v10 & (v51 + v33);
            v9 = (unsigned int)(v51 + 1);
            v34 = (int *)(*(_QWORD *)(a1 + 64) + 8LL * v33);
            v35 = *v34;
            if ( v32 == *v34 )
              goto LABEL_25;
            ++v51;
          }
          v49 = *(_DWORD *)(a1 + 72);
          if ( !v11 )
            v11 = (__int64)v34;
          ++*(_QWORD *)(a1 + 56);
          v50 = v49 + 1;
          v59 = v11;
          if ( 4 * v50 < 3 * v31 )
          {
            v9 = v31 >> 3;
            if ( v31 - *(_DWORD *)(a1 + 76) - v50 > (unsigned int)v9 )
            {
LABEL_44:
              *(_DWORD *)(a1 + 72) = v50;
              if ( *(_DWORD *)v11 != -1 )
                --*(_DWORD *)(a1 + 76);
              *(_DWORD *)v11 = v32;
              v12 = -2;
              v36 = 0;
              *(_DWORD *)(v11 + 4) = 0;
              goto LABEL_26;
            }
            v52 = v14;
LABEL_49:
            sub_2E518D0(v53, v31);
            sub_2E505D0(v53, &v57, &v59);
            v32 = v57;
            v14 = v52;
            v50 = *(_DWORD *)(a1 + 72) + 1;
            v11 = v59;
            goto LABEL_44;
          }
LABEL_48:
          v52 = v14;
          v31 *= 2;
          goto LABEL_49;
        }
LABEL_25:
        v11 = (unsigned int)v34[1];
        v12 = ~(1LL << v11);
        v36 = 8LL * ((unsigned int)v11 >> 6);
LABEL_26:
        *(_QWORD *)(v60[0] + v36) &= v12;
        if ( (*(_BYTE *)v13 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v13 + 44) & 8) != 0 )
            v13 = *(_QWORD *)(v13 + 8);
        }
        v13 = *(_QWORD *)(v13 + 8);
        if ( v14 == v13 )
          goto LABEL_10;
      }
      ++*(_QWORD *)(a1 + 56);
      v59 = 0;
      goto LABEL_48;
    }
LABEL_10:
    sub_2E4FCA0(v55 + 96, (__int64)v60, v12, v11, v9, v10);
    v15 = *(__int64 **)(a2 + 32);
    v16 = *(__int64 **)(a2 + 40);
    if ( v15 != v16 )
    {
      v17 = a2;
      v18 = &v59;
      v19 = a1 + 112;
      v20 = a1;
      v21 = v16;
      do
      {
        v27 = *(_QWORD *)(v20 + 8);
        v28 = *v15;
        v59 = *v15;
        v29 = *(_DWORD *)(v27 + 24);
        v30 = *(_QWORD *)(v27 + 8);
        if ( v29 )
        {
          v22 = v29 - 1;
          v23 = v22 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v24 = (__int64 *)(v30 + 16LL * v23);
          v25 = *v24;
          if ( v28 == *v24 )
          {
LABEL_13:
            if ( v28 != v58 )
            {
              v26 = v24[1];
              if ( v17 == v26 || v28 == **(_QWORD **)(v26 + 32) )
              {
                v54 = v17;
                v56 = v18;
                v37 = sub_2E57C80(v19, v18);
                sub_2E4FCA0(*v37 + 24, (__int64)v60, v38, v39, v40, v41);
                v42 = sub_2E57C80(v19, v56);
                sub_2E4FCA0(*v42 + 96, (__int64)v60, v43, v44, v45, v46);
                v17 = v54;
                v18 = v56;
              }
            }
            goto LABEL_16;
          }
          v47 = 1;
          while ( v25 != -4096 )
          {
            v48 = v47 + 1;
            v23 = v22 & (v47 + v23);
            v24 = (__int64 *)(v30 + 16LL * v23);
            v25 = *v24;
            if ( v28 == *v24 )
              goto LABEL_13;
            v47 = v48;
          }
        }
        if ( v28 != v58 )
          BUG();
LABEL_16:
        ++v15;
      }
      while ( v21 != v15 );
    }
    if ( (_BYTE *)v60[0] != v61 )
      _libc_free(v60[0]);
  }
}
