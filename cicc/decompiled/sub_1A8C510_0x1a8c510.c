// Function: sub_1A8C510
// Address: 0x1a8c510
//
void __fastcall sub_1A8C510(_QWORD *a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned int v7; // esi
  int v8; // edx
  __int64 v9; // rax
  _QWORD *v10; // rbx
  int v11; // ecx
  __int64 v12; // rdx
  unsigned __int64 *v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned int v16; // edx
  __int64 v17; // rcx
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  _QWORD *v20; // r13
  _QWORD *i; // rbx
  __int64 v22; // rdi
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rax
  int v32; // r8d
  unsigned int v33; // r9d
  __int64 *v34; // rdx
  __int64 v35; // r10
  __int64 *v36; // rax
  __int64 v37; // r14
  unsigned int v38; // r9d
  __int64 *v39; // rdx
  __int64 v40; // r11
  __int64 v41; // r12
  __int64 v42; // r15
  char *v43; // rdx
  char *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rsi
  char *v47; // rax
  __int64 v48; // rdx
  _BYTE *v49; // rsi
  int v50; // edx
  int v51; // r10d
  int v52; // edx
  int v53; // r11d
  _QWORD *v54; // r10
  int v55; // ecx
  int v56; // ecx
  int v57; // ecx
  int v58; // r10d
  _QWORD *v59; // r9
  __int64 v60; // rdi
  unsigned int v61; // edx
  __int64 v62; // rsi
  int v63; // edx
  int v64; // r10d
  __int64 v65; // rdi
  unsigned int v66; // ecx
  __int64 v67; // rsi
  int v68; // r11d
  __int64 v69; // [rsp+8h] [rbp-D8h]
  bool v70; // [rsp+27h] [rbp-B9h]
  _QWORD **v71; // [rsp+28h] [rbp-B8h]
  __int64 v72; // [rsp+30h] [rbp-B0h]
  __int64 v73; // [rsp+38h] [rbp-A8h]
  __int64 v74; // [rsp+40h] [rbp-A0h]
  __int64 v75; // [rsp+48h] [rbp-98h]
  int v76; // [rsp+50h] [rbp-90h]
  __int64 v78; // [rsp+60h] [rbp-80h]
  const char *v79; // [rsp+80h] [rbp-60h] BYREF
  __int64 v80; // [rsp+88h] [rbp-58h] BYREF
  __int64 v81; // [rsp+90h] [rbp-50h]
  __int64 v82; // [rsp+98h] [rbp-48h]
  __int64 v83; // [rsp+A0h] [rbp-40h]

  v69 = sub_13FC520(a1[7]);
  v1 = v69;
  v73 = sub_157F0B0(v69);
  v2 = sub_13FA090(a1[7]);
  v3 = a1[1];
  v72 = v2;
  v76 = a1[2] - 1;
  v71 = (_QWORD **)*a1;
  if ( *a1 != v3 )
  {
    v70 = v2 != -16 && v2 != -8 && v2 != 0;
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 + 8);
      LODWORD(v78) = v76;
      v79 = ".ldist";
      v5 = v4 + 224;
      v75 = v4 + 144;
      v80 = v78;
      LOWORD(v81) = 2307;
      v6 = sub_1AB91F0(v1, v73, *(_QWORD *)(v4 + 128), (int)v4 + 224, (unsigned int)&v79, a1[8], a1[9], v4 + 144);
      *(_QWORD *)(v4 + 136) = v6;
      v74 = v6;
      v80 = 2;
      v81 = 0;
      v82 = v72;
      if ( v70 )
        sub_164C220((__int64)&v80);
      v83 = v4 + 224;
      v79 = (const char *)&unk_49E6B50;
      v7 = *(_DWORD *)(v4 + 248);
      if ( !v7 )
        break;
      v9 = v82;
      v15 = *(_QWORD *)(v4 + 232);
      v16 = (v7 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
      v10 = (_QWORD *)(v15 + ((unsigned __int64)v16 << 6));
      v17 = v10[3];
      if ( v82 != v17 )
      {
        v53 = 1;
        v54 = 0;
        while ( v17 != -8 )
        {
          if ( v17 == -16 && !v54 )
            v54 = v10;
          v16 = (v7 - 1) & (v53 + v16);
          v10 = (_QWORD *)(v15 + ((unsigned __int64)v16 << 6));
          v17 = v10[3];
          if ( v82 == v17 )
            goto LABEL_21;
          ++v53;
        }
        v55 = *(_DWORD *)(v4 + 240);
        if ( v54 )
          v10 = v54;
        ++*(_QWORD *)(v4 + 224);
        v11 = v55 + 1;
        if ( 4 * v11 < 3 * v7 )
        {
          if ( v7 - *(_DWORD *)(v4 + 244) - v11 > v7 >> 3 )
            goto LABEL_10;
          sub_12E48B0(v4 + 224, v7);
          v56 = *(_DWORD *)(v4 + 248);
          if ( v56 )
          {
            v9 = v82;
            v57 = v56 - 1;
            v58 = 1;
            v59 = 0;
            v60 = *(_QWORD *)(v4 + 232);
            v61 = v57 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
            v10 = (_QWORD *)(v60 + ((unsigned __int64)v61 << 6));
            v62 = v10[3];
            if ( v82 != v62 )
            {
              while ( v62 != -8 )
              {
                if ( !v59 && v62 == -16 )
                  v59 = v10;
                v61 = v57 & (v61 + v58);
                v10 = (_QWORD *)(v60 + ((unsigned __int64)v61 << 6));
                v62 = v10[3];
                if ( v82 == v62 )
                  goto LABEL_9;
                ++v58;
              }
LABEL_103:
              if ( v59 )
                v10 = v59;
            }
LABEL_9:
            v11 = *(_DWORD *)(v4 + 240) + 1;
LABEL_10:
            *(_DWORD *)(v4 + 240) = v11;
            if ( v10[3] == -8 )
            {
              v13 = v10 + 1;
              if ( v9 != -8 )
                goto LABEL_15;
            }
            else
            {
              --*(_DWORD *)(v4 + 244);
              v12 = v10[3];
              if ( v9 != v12 )
              {
                v13 = v10 + 1;
                if ( v12 != -8 && v12 != 0 && v12 != -16 )
                {
                  sub_1649B30(v10 + 1);
                  v9 = v82;
                }
LABEL_15:
                v10[3] = v9;
                if ( v9 != 0 && v9 != -8 && v9 != -16 )
                  sub_1649AC0(v13, v80 & 0xFFFFFFFFFFFFFFF8LL);
                v9 = v82;
              }
            }
            v14 = v83;
            v10[5] = 6;
            v10[6] = 0;
            v10[4] = v14;
            v10[7] = 0;
            goto LABEL_21;
          }
LABEL_8:
          v9 = v82;
          v10 = 0;
          goto LABEL_9;
        }
LABEL_7:
        sub_12E48B0(v4 + 224, 2 * v7);
        v8 = *(_DWORD *)(v4 + 248);
        if ( v8 )
        {
          v9 = v82;
          v63 = v8 - 1;
          v64 = 1;
          v59 = 0;
          v65 = *(_QWORD *)(v4 + 232);
          v66 = v63 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
          v10 = (_QWORD *)(v65 + ((unsigned __int64)v66 << 6));
          v67 = v10[3];
          if ( v82 != v67 )
          {
            while ( v67 != -8 )
            {
              if ( !v59 && v67 == -16 )
                v59 = v10;
              v66 = v63 & (v66 + v64);
              v10 = (_QWORD *)(v65 + ((unsigned __int64)v66 << 6));
              v67 = v10[3];
              if ( v82 == v67 )
                goto LABEL_9;
              ++v64;
            }
            goto LABEL_103;
          }
          goto LABEL_9;
        }
        goto LABEL_8;
      }
LABEL_21:
      v79 = (const char *)&unk_49EE2B0;
      if ( v9 != 0 && v9 != -8 && v9 != -16 )
        sub_1649B30(&v80);
      v18 = v10[7];
      if ( v18 != v1 )
      {
        if ( v18 != 0 && v18 != -8 && v18 != -16 )
          sub_1649B30(v10 + 5);
        v10[7] = v1;
        if ( v1 != 0 && v1 != -8 && v1 != -16 )
          sub_164C220((__int64)(v10 + 5));
      }
      sub_1AB3E10(v75, v5);
      --v76;
      v3 = *(_QWORD *)(v3 + 8);
      v1 = sub_13FC520(v74);
      if ( v71 == (_QWORD **)v3 )
        goto LABEL_32;
    }
    ++*(_QWORD *)(v4 + 224);
    goto LABEL_7;
  }
  v1 = v69;
LABEL_32:
  v19 = sub_157EBA0(v73);
  sub_1648780(v19, v69, v1);
  v20 = (_QWORD *)*a1;
  for ( i = *(_QWORD **)*a1; a1 != i; i = (_QWORD *)*i )
  {
    v22 = v20[17];
    v23 = a1[9];
    if ( !v22 )
      v22 = v20[16];
    v24 = sub_13F9E70(v22);
    v25 = i[17];
    v26 = v24;
    if ( !v25 )
      v25 = i[16];
    v27 = sub_13FC520(v25);
    v29 = *(_QWORD *)(v23 + 32);
    v30 = v27;
    v31 = *(unsigned int *)(v23 + 48);
    if ( !(_DWORD)v31 )
      goto LABEL_119;
    v32 = v31 - 1;
    v33 = (v31 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v34 = (__int64 *)(v29 + 16LL * v33);
    v35 = *v34;
    if ( v26 == *v34 )
    {
LABEL_39:
      v36 = (__int64 *)(v29 + 16 * v31);
      if ( v36 != v34 )
      {
        v37 = v34[1];
        goto LABEL_41;
      }
    }
    else
    {
      v52 = 1;
      while ( v35 != -8 )
      {
        v68 = v52 + 1;
        v33 = v32 & (v52 + v33);
        v34 = (__int64 *)(v29 + 16LL * v33);
        v35 = *v34;
        if ( v26 == *v34 )
          goto LABEL_39;
        v52 = v68;
      }
      v36 = (__int64 *)(v29 + 16 * v31);
    }
    v37 = 0;
LABEL_41:
    v38 = v32 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
    v39 = (__int64 *)(v29 + 16LL * v38);
    v40 = *v39;
    if ( v30 != *v39 )
    {
      v50 = 1;
      while ( v40 != -8 )
      {
        v51 = v50 + 1;
        v38 = v32 & (v50 + v38);
        v39 = (__int64 *)(v29 + 16LL * v38);
        v40 = *v39;
        if ( v30 == *v39 )
          goto LABEL_42;
        v50 = v51;
      }
LABEL_119:
      *(_BYTE *)(v23 + 72) = 0;
      BUG();
    }
LABEL_42:
    if ( v36 == v39 )
      goto LABEL_119;
    v41 = v39[1];
    *(_BYTE *)(v23 + 72) = 0;
    v42 = *(_QWORD *)(v41 + 8);
    if ( v37 != v42 )
    {
      v43 = *(char **)(v42 + 32);
      v44 = *(char **)(v42 + 24);
      v45 = (v43 - v44) >> 5;
      v46 = (v43 - v44) >> 3;
      if ( v45 > 0 )
      {
        v47 = &v44[32 * v45];
        while ( v41 != *(_QWORD *)v44 )
        {
          if ( v41 == *((_QWORD *)v44 + 1) )
          {
            v44 += 8;
            goto LABEL_51;
          }
          if ( v41 == *((_QWORD *)v44 + 2) )
          {
            v44 += 16;
            goto LABEL_51;
          }
          if ( v41 == *((_QWORD *)v44 + 3) )
          {
            v44 += 24;
            goto LABEL_51;
          }
          v44 += 32;
          if ( v47 == v44 )
          {
            v46 = (v43 - v44) >> 3;
            goto LABEL_62;
          }
        }
        goto LABEL_51;
      }
LABEL_62:
      if ( v46 != 2 )
      {
        if ( v46 != 3 )
        {
          if ( v46 != 1 )
          {
            v44 = *(char **)(v42 + 32);
LABEL_51:
            if ( v44 + 8 != v43 )
            {
              memmove(v44, v44 + 8, v43 - (v44 + 8));
              v43 = *(char **)(v42 + 32);
            }
            v48 = (__int64)(v43 - 8);
            *(_QWORD *)(v42 + 32) = v48;
            *(_QWORD *)(v41 + 8) = v37;
            v79 = (const char *)v41;
            v49 = *(_BYTE **)(v37 + 32);
            if ( v49 == *(_BYTE **)(v37 + 40) )
            {
              sub_15CE310(v37 + 24, v49, &v79);
            }
            else
            {
              if ( v49 )
              {
                *(_QWORD *)v49 = v41;
                v49 = *(_BYTE **)(v37 + 32);
              }
              v49 += 8;
              *(_QWORD *)(v37 + 32) = v49;
            }
            if ( *(_DWORD *)(v41 + 16) != *(_DWORD *)(*(_QWORD *)(v41 + 8) + 16LL) + 1 )
              sub_1A89250(v41, (__int64)v49, v48, v28, v32, v38);
            goto LABEL_59;
          }
LABEL_74:
          if ( v41 != *(_QWORD *)v44 )
            v44 = *(char **)(v42 + 32);
          goto LABEL_51;
        }
        if ( v41 == *(_QWORD *)v44 )
          goto LABEL_51;
        v44 += 8;
      }
      if ( v41 == *(_QWORD *)v44 )
        goto LABEL_51;
      v44 += 8;
      goto LABEL_74;
    }
LABEL_59:
    v20 = (_QWORD *)*v20;
  }
}
