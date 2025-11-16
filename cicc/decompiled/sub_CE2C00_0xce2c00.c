// Function: sub_CE2C00
// Address: 0xce2c00
//
char __fastcall sub_CE2C00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // r10
  __int64 v7; // r14
  __int64 v8; // r10
  __int64 v9; // rsi
  int v10; // eax
  int v11; // ecx
  unsigned int v12; // eax
  __int64 v13; // rdx
  _QWORD *v14; // rbx
  _QWORD *v15; // r9
  __int64 v16; // r8
  __int64 v17; // r11
  _QWORD *v18; // r14
  __int64 v19; // r9
  unsigned int v20; // edi
  _QWORD *v21; // rax
  __int64 v22; // rcx
  _BYTE *v23; // r12
  __int64 v24; // rdx
  unsigned int v25; // r13d
  unsigned int v26; // esi
  _BYTE *v27; // rcx
  unsigned int v28; // esi
  int v29; // ecx
  int v30; // ecx
  __int64 v31; // rdi
  unsigned int v32; // r13d
  int v33; // eax
  _QWORD *v34; // rdx
  __int64 v35; // rsi
  int v36; // edi
  int v37; // edi
  unsigned int v38; // esi
  __int64 v39; // rdi
  unsigned int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // rcx
  int v43; // r12d
  __int64 *v44; // r9
  int v45; // eax
  int v46; // eax
  int v47; // ecx
  int v48; // ecx
  __int64 v49; // rdi
  _QWORD *v50; // r10
  unsigned int v51; // r13d
  int v52; // r9d
  _BYTE *v53; // rsi
  int v54; // r9d
  int v55; // ebx
  int v56; // ebx
  __int64 v57; // r8
  unsigned int v58; // edx
  __int64 v59; // rdi
  int v60; // esi
  __int64 *v61; // rcx
  int v62; // eax
  __int64 v63; // rdi
  int v64; // r12d
  int v65; // ecx
  unsigned int v66; // ebx
  __int64 *v67; // rdx
  __int64 v68; // rsi
  __int64 v70; // [rsp+0h] [rbp-60h]
  __int64 v71; // [rsp+0h] [rbp-60h]
  __int64 v72; // [rsp+8h] [rbp-58h]
  int v73; // [rsp+8h] [rbp-58h]
  __int64 v74; // [rsp+8h] [rbp-58h]
  __int64 v75; // [rsp+8h] [rbp-58h]
  __int64 v76; // [rsp+8h] [rbp-58h]
  __int64 v78; // [rsp+18h] [rbp-48h]
  __int64 v80; // [rsp+28h] [rbp-38h]

  LOBYTE(v4) = a2 + 48;
  v5 = *(_QWORD *)(a2 + 56);
  v80 = a2 + 48;
  if ( v5 == a2 + 48 )
    return v4;
LABEL_4:
  while ( 2 )
  {
    v7 = v5 - 24;
    if ( !v5 )
      v7 = 0;
    LOBYTE(v4) = sub_CE16D0(v7);
    if ( (_BYTE)v4 )
    {
      v9 = *(_QWORD *)(a1 + 64);
      v10 = *(_DWORD *)(a1 + 80);
      if ( v10 )
      {
        v11 = v10 - 1;
        v12 = (v10 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v13 = *(_QWORD *)(v9 + 16LL * v12);
        if ( v7 == v13 )
        {
LABEL_9:
          v4 = 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
          if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
          {
            v14 = *(_QWORD **)(v7 - 8);
            v15 = (_QWORD *)((char *)v14 + v4);
          }
          else
          {
            v15 = (_QWORD *)v7;
            v14 = (_QWORD *)(v7 - v4);
          }
          if ( v14 == v15 )
            goto LABEL_3;
          v16 = v7;
          v78 = v8;
          v17 = a4;
          v18 = v15;
          while ( 1 )
          {
            v23 = (_BYTE *)*v14;
            if ( *(_BYTE *)*v14 <= 0x1Cu )
              goto LABEL_16;
            LODWORD(v4) = *(_DWORD *)(v17 + 24);
            v24 = *(_QWORD *)(v17 + 8);
            if ( !(_DWORD)v4 )
              goto LABEL_16;
            LODWORD(v4) = v4 - 1;
            v25 = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
            v26 = v4 & v25;
            v27 = *(_BYTE **)(v24 + 8LL * ((unsigned int)v4 & v25));
            if ( v23 != v27 )
            {
              v36 = 1;
              while ( v27 != (_BYTE *)-4096LL )
              {
                v26 = v4 & (v36 + v26);
                v27 = *(_BYTE **)(v24 + 8LL * v26);
                if ( v23 == v27 )
                  goto LABEL_20;
                ++v36;
              }
              goto LABEL_16;
            }
LABEL_20:
            v28 = *(_DWORD *)(a3 + 24);
            if ( !v28 )
              break;
            v19 = *(_QWORD *)(a3 + 8);
            v20 = (v28 - 1) & v25;
            v21 = (_QWORD *)(v19 + 16LL * v20);
            v22 = *v21;
            if ( v23 != (_BYTE *)*v21 )
            {
              v73 = 1;
              v34 = 0;
              while ( v22 != -4096 )
              {
                if ( v22 == -8192 && !v34 )
                  v34 = v21;
                v20 = (v28 - 1) & (v73 + v20);
                v21 = (_QWORD *)(v19 + 16LL * v20);
                v22 = *v21;
                if ( v23 == (_BYTE *)*v21 )
                  goto LABEL_14;
                ++v73;
              }
              if ( !v34 )
                v34 = v21;
              v46 = *(_DWORD *)(a3 + 16);
              ++*(_QWORD *)a3;
              v33 = v46 + 1;
              if ( 4 * v33 < 3 * v28 )
              {
                if ( v28 - *(_DWORD *)(a3 + 20) - v33 <= v28 >> 3 )
                {
                  v71 = v17;
                  v74 = v16;
                  sub_CE25F0(a3, v28);
                  v47 = *(_DWORD *)(a3 + 24);
                  if ( !v47 )
                  {
LABEL_108:
                    ++*(_DWORD *)(a3 + 16);
                    BUG();
                  }
                  v48 = v47 - 1;
                  v49 = *(_QWORD *)(a3 + 8);
                  v50 = 0;
                  v51 = v48 & v25;
                  v16 = v74;
                  v17 = v71;
                  v52 = 1;
                  v33 = *(_DWORD *)(a3 + 16) + 1;
                  v34 = (_QWORD *)(v49 + 16LL * v51);
                  v53 = (_BYTE *)*v34;
                  if ( v23 != (_BYTE *)*v34 )
                  {
                    while ( v53 != (_BYTE *)-4096LL )
                    {
                      if ( !v50 && v53 == (_BYTE *)-8192LL )
                        v50 = v34;
                      v51 = v48 & (v52 + v51);
                      v34 = (_QWORD *)(v49 + 16LL * v51);
                      v53 = (_BYTE *)*v34;
                      if ( v23 == (_BYTE *)*v34 )
                        goto LABEL_24;
                      ++v52;
                    }
LABEL_68:
                    if ( v50 )
                      v34 = v50;
                  }
                }
LABEL_24:
                *(_DWORD *)(a3 + 16) = v33;
                if ( *v34 != -4096 )
                  --*(_DWORD *)(a3 + 20);
                *v34 = v23;
                v4 = (__int64)(v34 + 1);
                v34[1] = 0;
                goto LABEL_15;
              }
LABEL_22:
              v70 = v17;
              v72 = v16;
              sub_CE25F0(a3, 2 * v28);
              v29 = *(_DWORD *)(a3 + 24);
              if ( !v29 )
                goto LABEL_108;
              v30 = v29 - 1;
              v31 = *(_QWORD *)(a3 + 8);
              v32 = v30 & v25;
              v16 = v72;
              v17 = v70;
              v33 = *(_DWORD *)(a3 + 16) + 1;
              v34 = (_QWORD *)(v31 + 16LL * v32);
              v35 = *v34;
              if ( v23 != (_BYTE *)*v34 )
              {
                v54 = 1;
                v50 = 0;
                while ( v35 != -4096 )
                {
                  if ( v35 == -8192 && !v50 )
                    v50 = v34;
                  v32 = v30 & (v54 + v32);
                  v34 = (_QWORD *)(v31 + 16LL * v32);
                  v35 = *v34;
                  if ( v23 == (_BYTE *)*v34 )
                    goto LABEL_24;
                  ++v54;
                }
                goto LABEL_68;
              }
              goto LABEL_24;
            }
LABEL_14:
            v4 = (__int64)(v21 + 1);
LABEL_15:
            *(_QWORD *)v4 = v16;
LABEL_16:
            v14 += 4;
            if ( v18 == v14 )
            {
              v5 = *(_QWORD *)(v78 + 8);
              if ( v80 == v5 )
                return v4;
              goto LABEL_4;
            }
          }
          ++*(_QWORD *)a3;
          goto LABEL_22;
        }
        v37 = 1;
        while ( v13 != -4096 )
        {
          v12 = v11 & (v37 + v12);
          v13 = *(_QWORD *)(v9 + 16LL * v12);
          if ( v7 == v13 )
            goto LABEL_9;
          ++v37;
        }
      }
      v38 = *(_DWORD *)(a4 + 24);
      if ( v38 )
      {
        v39 = *(_QWORD *)(a4 + 8);
        v40 = (v38 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v41 = (__int64 *)(v39 + 8LL * v40);
        v42 = *v41;
        if ( v7 == *v41 )
          goto LABEL_9;
        v43 = 1;
        v44 = 0;
        while ( v42 != -4096 )
        {
          if ( v44 || v42 != -8192 )
            v41 = v44;
          v40 = (v38 - 1) & (v43 + v40);
          v42 = *(_QWORD *)(v39 + 8LL * v40);
          if ( v7 == v42 )
            goto LABEL_9;
          ++v43;
          v44 = v41;
          v41 = (__int64 *)(v39 + 8LL * v40);
        }
        if ( !v44 )
          v44 = v41;
        ++*(_QWORD *)a4;
        v45 = *(_DWORD *)(a4 + 16) + 1;
        if ( 4 * v45 < 3 * v38 )
        {
          if ( v38 - *(_DWORD *)(a4 + 20) - v45 <= v38 >> 3 )
          {
            v76 = v8;
            sub_CE2A30(a4, v38);
            v62 = *(_DWORD *)(a4 + 24);
            if ( !v62 )
            {
LABEL_107:
              ++*(_DWORD *)(a4 + 16);
              BUG();
            }
            v63 = *(_QWORD *)(a4 + 8);
            v64 = v62 - 1;
            v8 = v76;
            v65 = 1;
            v66 = (v62 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v44 = (__int64 *)(v63 + 8LL * v66);
            v67 = 0;
            v68 = *v44;
            v45 = *(_DWORD *)(a4 + 16) + 1;
            if ( v7 != *v44 )
            {
              while ( v68 != -4096 )
              {
                if ( !v67 && v68 == -8192 )
                  v67 = v44;
                v66 = v64 & (v65 + v66);
                v44 = (__int64 *)(v63 + 8LL * v66);
                v68 = *v44;
                if ( v7 == *v44 )
                  goto LABEL_44;
                ++v65;
              }
              if ( v67 )
                v44 = v67;
            }
          }
          goto LABEL_44;
        }
      }
      else
      {
        ++*(_QWORD *)a4;
      }
      v75 = v8;
      sub_CE2A30(a4, 2 * v38);
      v55 = *(_DWORD *)(a4 + 24);
      if ( !v55 )
        goto LABEL_107;
      v56 = v55 - 1;
      v57 = *(_QWORD *)(a4 + 8);
      v8 = v75;
      v58 = v56 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v44 = (__int64 *)(v57 + 8LL * v58);
      v59 = *v44;
      v45 = *(_DWORD *)(a4 + 16) + 1;
      if ( v7 != *v44 )
      {
        v60 = 1;
        v61 = 0;
        while ( v59 != -4096 )
        {
          if ( v59 == -8192 && !v61 )
            v61 = v44;
          v58 = v56 & (v60 + v58);
          v44 = (__int64 *)(v57 + 8LL * v58);
          v59 = *v44;
          if ( v7 == *v44 )
            goto LABEL_44;
          ++v60;
        }
        if ( v61 )
          v44 = v61;
      }
LABEL_44:
      *(_DWORD *)(a4 + 16) = v45;
      if ( *v44 != -4096 )
        --*(_DWORD *)(a4 + 20);
      *v44 = v7;
      goto LABEL_9;
    }
LABEL_3:
    v5 = *(_QWORD *)(v8 + 8);
    if ( v80 != v5 )
      continue;
    return v4;
  }
}
