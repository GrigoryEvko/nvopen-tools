// Function: sub_2B826B0
// Address: 0x2b826b0
//
__int64 __fastcall sub_2B826B0(__int64 *a1)
{
  __int64 v1; // rax
  signed int v2; // ebx
  unsigned int v4; // r15d
  unsigned int v5; // r12d
  unsigned __int64 v6; // rax
  unsigned int v7; // r14d
  int v8; // edx
  _DWORD *v9; // rdi
  unsigned int i; // esi
  _DWORD *v11; // rax
  __int64 v12; // r8
  int v13; // r12d
  unsigned int v14; // r10d
  int *v15; // rax
  int v16; // edx
  __int64 v17; // rax
  _BYTE *v18; // rax
  _BYTE *v19; // rdi
  unsigned int v20; // edx
  unsigned int v21; // r12d
  unsigned int v23; // r14d
  __int64 v24; // rax
  char v25; // dl
  unsigned int v26; // edx
  unsigned int v27; // eax
  int v28; // eax
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r12
  _DWORD *v32; // rax
  _DWORD *v33; // rdx
  unsigned int v34; // r15d
  _DWORD *v35; // rax
  int *v36; // r9
  int v37; // eax
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  _BYTE *v40; // rax
  __int64 v41; // rcx
  int v42; // r10d
  int *v43; // rdi
  int *v44; // rdi
  int v45; // r11d
  unsigned int v46; // ecx
  int v47; // esi
  _DWORD *v48; // r9
  unsigned int v49; // [rsp+0h] [rbp-90h]
  int *v50; // [rsp+0h] [rbp-90h]
  int v51; // [rsp+0h] [rbp-90h]
  int v52; // [rsp+Ch] [rbp-84h]
  int v53; // [rsp+Ch] [rbp-84h]
  int v54; // [rsp+Ch] [rbp-84h]
  int v55; // [rsp+Ch] [rbp-84h]
  unsigned int v56; // [rsp+Ch] [rbp-84h]
  unsigned int v57; // [rsp+14h] [rbp-7Ch] BYREF
  unsigned __int64 v58; // [rsp+18h] [rbp-78h]
  int v59; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v60[12]; // [rsp+24h] [rbp-6Ch] BYREF
  __int64 v61; // [rsp+30h] [rbp-60h] BYREF
  _DWORD *v62; // [rsp+38h] [rbp-58h]
  __int64 v63; // [rsp+40h] [rbp-50h]
  unsigned int v64; // [rsp+48h] [rbp-48h]
  _BYTE *v65; // [rsp+50h] [rbp-40h] BYREF
  __int64 v66; // [rsp+58h] [rbp-38h]
  _BYTE v67[48]; // [rsp+60h] [rbp-30h] BYREF

  v65 = v67;
  v1 = *a1;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v66 = 0;
  v2 = *(_DWORD *)(v1 + 8);
  v61 = 0;
  if ( v2 <= 0 )
  {
    v21 = 0;
    goto LABEL_24;
  }
  v4 = 0;
  v5 = -1;
  do
  {
    while ( 1 )
    {
      v57 = --v2;
      v6 = sub_2B68480((__int64)a1, v2);
      v58 = v6;
      v7 = v6;
      v59 = v8;
      if ( (unsigned int)v6 >= v5 )
        break;
      ++v61;
      v4 = HIDWORD(v58);
      v9 = v62;
      i = v64;
      if ( !(_DWORD)v63 )
      {
        if ( !HIDWORD(v63) )
          goto LABEL_13;
        if ( v64 > 0x40 )
        {
          sub_C7D6A0((__int64)v62, 8LL * v64, 4);
          v64 = 0;
          LODWORD(v12) = v57;
          v62 = 0;
          v13 = v59;
          v63 = 0;
          LODWORD(v66) = 0;
LABEL_55:
          ++v61;
          i = 0;
          v54 = v13;
          goto LABEL_56;
        }
LABEL_9:
        v11 = &v62[2 * v64];
        if ( v11 != v62 )
        {
          do
          {
            *v9 = -1;
            v9 += 2;
          }
          while ( v11 != v9 );
          v9 = v62;
          i = v64;
        }
        v63 = 0;
        goto LABEL_13;
      }
      v26 = 4 * v63;
      if ( (unsigned int)(4 * v63) < 0x40 )
        v26 = 64;
      if ( v26 >= v64 )
        goto LABEL_9;
      if ( (_DWORD)v63 == 1 )
      {
        v52 = 128;
        v31 = 1024;
LABEL_37:
        sub_C7D6A0((__int64)v62, 8LL * v64, 4);
        v64 = v52;
        v32 = (_DWORD *)sub_C7D670(v31, 4);
        v63 = 0;
        v9 = v32;
        v62 = v32;
        v33 = &v32[2 * v64];
        for ( i = v64; v33 != v32; v32 += 2 )
        {
          if ( v32 )
            *v32 = -1;
        }
        goto LABEL_13;
      }
      _BitScanReverse(&v27, v63 - 1);
      v28 = 1 << (33 - (v27 ^ 0x1F));
      if ( v28 < 64 )
        v28 = 64;
      if ( v28 != v64 )
      {
        v29 = (4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1);
        v30 = ((((v29 >> 2) | v29 | (((v29 >> 2) | v29) >> 4)) >> 8)
             | (v29 >> 2)
             | v29
             | (((v29 >> 2) | v29) >> 4)
             | (((((v29 >> 2) | v29 | (((v29 >> 2) | v29) >> 4)) >> 8) | (v29 >> 2) | v29 | (((v29 >> 2) | v29) >> 4)) >> 16))
            + 1;
        v52 = v30;
        v31 = 8 * v30;
        goto LABEL_37;
      }
      v63 = 0;
      v48 = &v62[2 * v64];
      do
      {
        if ( v9 )
          *v9 = -1;
        v9 += 2;
      }
      while ( v48 != v9 );
      v9 = v62;
      i = v64;
LABEL_13:
      v12 = v57;
      v13 = v59;
      LODWORD(v66) = 0;
      if ( !i )
        goto LABEL_55;
      v14 = (i - 1) & (37 * v59);
      v15 = &v9[2 * v14];
      v16 = *v15;
      if ( v59 != *v15 )
      {
        v53 = 1;
        v36 = 0;
        while ( v16 != -1 )
        {
          if ( !v36 && v16 == -2 )
            v36 = v15;
          v14 = (i - 1) & (v53 + v14);
          v15 = &v9[2 * v14];
          v16 = *v15;
          if ( v59 == *v15 )
            goto LABEL_15;
          ++v53;
        }
        if ( !v36 )
          v36 = v15;
        ++v61;
        v37 = v63 + 1;
        if ( 4 * ((int)v63 + 1) < 3 * i )
        {
          if ( i - (v37 + HIDWORD(v63)) <= i >> 3 )
          {
            v51 = 37 * v59;
            v56 = v57;
            sub_A09770((__int64)&v61, i);
            if ( !v64 )
            {
LABEL_93:
              LODWORD(v63) = v63 + 1;
              BUG();
            }
            v44 = 0;
            v12 = v56;
            v45 = 1;
            v46 = (v64 - 1) & v51;
            v36 = &v62[2 * v46];
            v47 = *v36;
            v37 = v63 + 1;
            if ( v13 != *v36 )
            {
              while ( v47 != -1 )
              {
                if ( !v44 && v47 == -2 )
                  v44 = v36;
                v46 = (v64 - 1) & (v45 + v46);
                v36 = &v62[2 * v46];
                v47 = *v36;
                if ( v13 == *v36 )
                  goto LABEL_49;
                ++v45;
              }
              if ( v44 )
                v36 = v44;
            }
          }
          goto LABEL_49;
        }
        v54 = v59;
LABEL_56:
        v49 = v12;
        sub_A09770((__int64)&v61, 2 * i);
        if ( !v64 )
          goto LABEL_93;
        v12 = v49;
        LODWORD(v41) = (v64 - 1) & (37 * v13);
        v36 = &v62[2 * (unsigned int)v41];
        v13 = *v36;
        v37 = v63 + 1;
        if ( *v36 != v54 )
        {
          v42 = 1;
          v43 = 0;
          while ( v13 != -1 )
          {
            if ( v13 == -2 && !v43 )
              v43 = v36;
            v41 = (v64 - 1) & ((_DWORD)v41 + v42);
            v36 = &v62[2 * v41];
            v13 = *v36;
            if ( *v36 == v54 )
              goto LABEL_49;
            ++v42;
          }
          v13 = v54;
          if ( v43 )
            v36 = v43;
        }
LABEL_49:
        LODWORD(v63) = v37;
        if ( *v36 != -1 )
          --HIDWORD(v63);
        *v36 = v13;
        v36[1] = 0;
        *(_QWORD *)&v60[4] = 0;
        *(_DWORD *)v60 = v59;
        v38 = (unsigned int)v66;
        v39 = (unsigned int)v66 + 1LL;
        if ( v39 > HIDWORD(v66) )
        {
          v50 = v36;
          v55 = v12;
          sub_C8D5F0((__int64)&v65, v67, v39, 0xCu, v12, (__int64)v36);
          v38 = (unsigned int)v66;
          v36 = v50;
          LODWORD(v12) = v55;
        }
        v40 = &v65[12 * v38];
        *(_QWORD *)v40 = *(_QWORD *)v60;
        *((_DWORD *)v40 + 2) = *(_DWORD *)&v60[8];
        v17 = (unsigned int)v66;
        LODWORD(v66) = v66 + 1;
        v36[1] = v17;
        goto LABEL_16;
      }
LABEL_15:
      v17 = (unsigned int)v15[1];
LABEL_16:
      v5 = v7;
      v18 = &v65[12 * v17];
      *((_DWORD *)v18 + 1) = 1;
      *((_DWORD *)v18 + 2) = v12;
      if ( !v2 )
        goto LABEL_17;
    }
    if ( (_DWORD)v6 == v5 )
    {
      v23 = HIDWORD(v58);
      if ( HIDWORD(v58) < v4 )
      {
        v34 = v57;
        v35 = (_DWORD *)sub_2B82420((__int64)&v61, (unsigned int *)&v59);
        v35[1] = v34;
        v4 = v23;
        *v35 = 1;
      }
      else if ( HIDWORD(v58) == v4 )
      {
        *(_DWORD *)v60 = 1;
        v24 = sub_2B82130((__int64)&v61, &v59, v60, (int *)&v57);
        if ( !v25 )
          ++*(_DWORD *)(v24 + 4);
      }
    }
  }
  while ( v2 );
LABEL_17:
  v19 = &v65[12 * (unsigned int)v66];
  if ( v65 == v19 )
  {
    v21 = 0;
  }
  else
  {
    v20 = -1;
    v21 = 0;
    do
    {
      if ( *((_DWORD *)v19 - 2) < v20 )
      {
        v21 = *((_DWORD *)v19 - 1);
        v20 = *((_DWORD *)v19 - 2);
      }
      v19 -= 12;
    }
    while ( v65 != v19 );
  }
  if ( v19 != v67 )
    _libc_free((unsigned __int64)v19);
LABEL_24:
  sub_C7D6A0((__int64)v62, 8LL * v64, 4);
  return v21;
}
