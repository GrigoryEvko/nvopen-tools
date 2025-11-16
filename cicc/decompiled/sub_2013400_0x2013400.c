// Function: sub_2013400
// Address: 0x2013400
//
void __fastcall sub_2013400(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __m128i *a5, const __m128i *a6)
{
  __int64 *v7; // rax
  bool v8; // zf
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // r12d
  int v13; // eax
  char v14; // cl
  __int64 v15; // r8
  int v16; // esi
  unsigned int v17; // edi
  _DWORD *v18; // rdx
  int v19; // r9d
  __m128i *v20; // r8
  __int64 v21; // r9
  int v22; // eax
  __int64 v23; // rcx
  unsigned __int64 v24; // r12
  _QWORD *v25; // rdi
  int v26; // esi
  __int64 v27; // rdx
  __int64 *v28; // r15
  unsigned int v29; // r14d
  int v30; // esi
  unsigned int v31; // edi
  unsigned int *v32; // rdx
  unsigned int v33; // r10d
  unsigned int v34; // r13d
  unsigned int v35; // eax
  char v36; // cl
  unsigned int v37; // esi
  unsigned int v38; // edi
  int v39; // r8d
  unsigned int v40; // r11d
  unsigned int v41; // ecx
  _BYTE *v42; // rdi
  int v43; // r11d
  unsigned int v44; // esi
  __int64 v45; // rdi
  int v46; // ecx
  unsigned int v47; // esi
  __int64 v48; // rdi
  int v49; // ecx
  unsigned int v50; // esi
  unsigned int *v51; // r10
  int v52; // ecx
  int v53; // ecx
  unsigned int v54; // edi
  int v55; // r8d
  unsigned int v56; // r10d
  int v57; // r13d
  _DWORD *v58; // r10
  int v59; // esi
  __int64 v60; // r9
  int v61; // esi
  unsigned int v62; // ecx
  int v63; // r11d
  __int64 v64; // r9
  int v65; // esi
  unsigned int v66; // ecx
  int v67; // r11d
  int v68; // r8d
  _DWORD *v69; // rdi
  int v70; // esi
  int v71; // r8d
  int v74; // [rsp+28h] [rbp-1A8h]
  unsigned int v75; // [rsp+2Ch] [rbp-1A4h]
  int v76; // [rsp+2Ch] [rbp-1A4h]
  int v77; // [rsp+2Ch] [rbp-1A4h]
  unsigned __int64 v78; // [rsp+30h] [rbp-1A0h] BYREF
  __m128i *v79; // [rsp+38h] [rbp-198h]
  unsigned __int64 v80; // [rsp+40h] [rbp-190h] BYREF
  __int64 v81; // [rsp+48h] [rbp-188h]
  __int64 (__fastcall **v82)(); // [rsp+50h] [rbp-180h] BYREF
  __int64 v83; // [rsp+58h] [rbp-178h]
  __int64 v84; // [rsp+60h] [rbp-170h]
  __int64 v85; // [rsp+68h] [rbp-168h]
  __int64 *v86; // [rsp+70h] [rbp-160h]
  __int64 v87; // [rsp+80h] [rbp-150h] BYREF
  __int64 v88; // [rsp+88h] [rbp-148h]
  _QWORD *v89; // [rsp+90h] [rbp-140h] BYREF
  unsigned int v90; // [rsp+98h] [rbp-138h]
  _BYTE *v91; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v92; // [rsp+118h] [rbp-B8h]
  _BYTE v93[176]; // [rsp+120h] [rbp-B0h] BYREF

  v78 = a4;
  v79 = a5;
  v7 = sub_2010420(a1, a4, a3, a4, a5, a6);
  v8 = *((_DWORD *)v7 + 7) == -3;
  v78 = (unsigned __int64)v7;
  if ( v8 )
    sub_2010110(a1, (__int64)&v78);
  v9 = &v89;
  v87 = 0;
  v88 = 1;
  do
    *v9++ = -8;
  while ( v9 != &v91 );
  v85 = a1;
  v91 = v93;
  v92 = 0x1000000000LL;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)(v10 + 664);
  v84 = v10;
  v83 = v11;
  *(_QWORD *)(v10 + 664) = &v82;
  v82 = off_4985988;
  v86 = &v87;
  do
  {
    v12 = sub_200F8F0(a1, a2, a3);
    v13 = sub_200F8F0(a1, v78, (__int64)v79);
    if ( v12 == v13 )
      goto LABEL_11;
    v14 = *(_BYTE *)(a1 + 1296) & 1;
    if ( v14 )
    {
      v15 = a1 + 1304;
      v16 = 7;
    }
    else
    {
      v44 = *(_DWORD *)(a1 + 1312);
      v15 = *(_QWORD *)(a1 + 1304);
      if ( !v44 )
      {
        v54 = *(_DWORD *)(a1 + 1296);
        ++*(_QWORD *)(a1 + 1288);
        v18 = 0;
        v55 = (v54 >> 1) + 1;
        goto LABEL_75;
      }
      v16 = v44 - 1;
    }
    v17 = v16 & (37 * v12);
    v18 = (_DWORD *)(v15 + 8LL * v17);
    v19 = *v18;
    if ( v12 != *v18 )
    {
      v57 = 1;
      v58 = 0;
      while ( v19 != -1 )
      {
        if ( !v58 && v19 == -2 )
          v58 = v18;
        v17 = v16 & (v57 + v17);
        v18 = (_DWORD *)(v15 + 8LL * v17);
        v19 = *v18;
        if ( v12 == *v18 )
          goto LABEL_10;
        ++v57;
      }
      v54 = *(_DWORD *)(a1 + 1296);
      v44 = 8;
      if ( v58 )
        v18 = v58;
      ++*(_QWORD *)(a1 + 1288);
      v56 = 24;
      v55 = (v54 >> 1) + 1;
      if ( v14 )
      {
LABEL_76:
        if ( 4 * v55 >= v56 )
        {
          v76 = v13;
          sub_20108A0(a1 + 1288, 2 * v44);
          v13 = v76;
          if ( (*(_BYTE *)(a1 + 1296) & 1) != 0 )
          {
            v60 = a1 + 1304;
            v61 = 7;
          }
          else
          {
            v59 = *(_DWORD *)(a1 + 1312);
            v60 = *(_QWORD *)(a1 + 1304);
            if ( !v59 )
              goto LABEL_136;
            v61 = v59 - 1;
          }
          v62 = v61 & (37 * v12);
          v18 = (_DWORD *)(v60 + 8LL * v62);
          v63 = *v18;
          if ( v12 == *v18 )
            goto LABEL_96;
          v71 = 1;
          v69 = 0;
          while ( v63 != -1 )
          {
            if ( !v69 && v63 == -2 )
              v69 = v18;
            v62 = v61 & (v71 + v62);
            v18 = (_DWORD *)(v60 + 8LL * v62);
            v63 = *v18;
            if ( v12 == *v18 )
              goto LABEL_96;
            ++v71;
          }
        }
        else
        {
          if ( v44 - *(_DWORD *)(a1 + 1300) - v55 > v44 >> 3 )
          {
LABEL_78:
            *(_DWORD *)(a1 + 1296) = (2 * (v54 >> 1) + 2) | v54 & 1;
            if ( *v18 != -1 )
              --*(_DWORD *)(a1 + 1300);
            *v18 = v12;
            v18[1] = 0;
            goto LABEL_10;
          }
          v77 = v13;
          sub_20108A0(a1 + 1288, v44);
          v13 = v77;
          if ( (*(_BYTE *)(a1 + 1296) & 1) != 0 )
          {
            v64 = a1 + 1304;
            v65 = 7;
          }
          else
          {
            v70 = *(_DWORD *)(a1 + 1312);
            v64 = *(_QWORD *)(a1 + 1304);
            if ( !v70 )
            {
LABEL_136:
              *(_DWORD *)(a1 + 1296) = (2 * (*(_DWORD *)(a1 + 1296) >> 1) + 2) | *(_DWORD *)(a1 + 1296) & 1;
              BUG();
            }
            v65 = v70 - 1;
          }
          v66 = v65 & (37 * v12);
          v18 = (_DWORD *)(v64 + 8LL * v66);
          v67 = *v18;
          if ( v12 == *v18 )
          {
LABEL_96:
            v54 = *(_DWORD *)(a1 + 1296);
            goto LABEL_78;
          }
          v68 = 1;
          v69 = 0;
          while ( v67 != -1 )
          {
            if ( v67 == -2 && !v69 )
              v69 = v18;
            v66 = v65 & (v68 + v66);
            v18 = (_DWORD *)(v64 + 8LL * v66);
            v67 = *v18;
            if ( v12 == *v18 )
              goto LABEL_96;
            ++v68;
          }
        }
        if ( v69 )
          v18 = v69;
        goto LABEL_96;
      }
      v44 = *(_DWORD *)(a1 + 1312);
LABEL_75:
      v56 = 3 * v44;
      goto LABEL_76;
    }
LABEL_10:
    v18[1] = v13;
LABEL_11:
    sub_1D44C70(*(_QWORD *)(a1 + 8), a2, a3, v78, (unsigned int)v79);
LABEL_12:
    v22 = v92;
    while ( v22 )
    {
      v23 = (__int64)v91;
      v24 = *(_QWORD *)&v91[8 * v22 - 8];
      if ( (v88 & 1) != 0 )
      {
        v25 = &v89;
        v26 = 15;
LABEL_16:
        v27 = v26 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v23 = (__int64)&v25[v27];
        v20 = *(__m128i **)v23;
        if ( v24 == *(_QWORD *)v23 )
        {
LABEL_17:
          *(_QWORD *)v23 = -16;
          ++HIDWORD(v88);
          v27 = 2 * ((unsigned int)v88 >> 1) - 2;
          LODWORD(v88) = v27 | v88 & 1;
          v22 = v92;
        }
        else
        {
          v23 = 1;
          while ( v20 != (__m128i *)-8LL )
          {
            v21 = (unsigned int)(v23 + 1);
            v27 = v26 & (unsigned int)(v23 + v27);
            v23 = (__int64)&v25[(unsigned int)v27];
            v20 = *(__m128i **)v23;
            if ( v24 == *(_QWORD *)v23 )
              goto LABEL_17;
            v23 = (unsigned int)v21;
          }
        }
        goto LABEL_18;
      }
      v27 = v90;
      v25 = v89;
      v26 = v90 - 1;
      if ( v90 )
        goto LABEL_16;
LABEL_18:
      LODWORD(v92) = --v22;
      if ( *(_DWORD *)(v24 + 28) == -1 )
      {
        v28 = sub_2010420(a1, v24, v27, v23, v20, (const __m128i *)v21);
        if ( (__int64 *)v24 != v28 )
        {
          v29 = 0;
          v74 = *(_DWORD *)(v24 + 60);
          if ( v74 )
          {
            while ( 2 )
            {
              LODWORD(v81) = v29;
              v8 = *((_DWORD *)v28 + 7) == -3;
              v80 = (unsigned __int64)v28;
              if ( v8 )
                sub_2010110(a1, (__int64)&v80);
              v34 = sub_200F8F0(a1, v24, v29);
              v75 = sub_200F8F0(a1, v80, v81);
              sub_1D44C70(*(_QWORD *)(a1 + 8), v24, v29, v80, v81);
              v35 = v75;
              if ( v34 == v75 )
                goto LABEL_25;
              v36 = *(_BYTE *)(a1 + 1296) & 1;
              if ( v36 )
              {
                v20 = (__m128i *)(a1 + 1304);
                v30 = 7;
                goto LABEL_23;
              }
              v37 = *(_DWORD *)(a1 + 1312);
              v20 = *(__m128i **)(a1 + 1304);
              if ( !v37 )
              {
                v38 = *(_DWORD *)(a1 + 1296);
                ++*(_QWORD *)(a1 + 1288);
                v32 = 0;
                v39 = (v38 >> 1) + 1;
                goto LABEL_35;
              }
              v30 = v37 - 1;
LABEL_23:
              v31 = v30 & (37 * v34);
              v32 = (unsigned int *)v20 + 2 * v31;
              v33 = *v32;
              if ( v34 == *v32 )
              {
LABEL_24:
                v32[1] = v35;
LABEL_25:
                if ( ++v29 == v74 )
                  goto LABEL_12;
                continue;
              }
              break;
            }
            v43 = 1;
            v21 = 0;
            while ( v33 != -1 )
            {
              if ( !v21 && v33 == -2 )
                v21 = (__int64)v32;
              v31 = v30 & (v43 + v31);
              v32 = (unsigned int *)v20 + 2 * v31;
              v33 = *v32;
              if ( v34 == *v32 )
                goto LABEL_24;
              ++v43;
            }
            v38 = *(_DWORD *)(a1 + 1296);
            v40 = 24;
            v37 = 8;
            if ( v21 )
              v32 = (unsigned int *)v21;
            ++*(_QWORD *)(a1 + 1288);
            v39 = (v38 >> 1) + 1;
            if ( !v36 )
            {
              v37 = *(_DWORD *)(a1 + 1312);
LABEL_35:
              v40 = 3 * v37;
            }
            if ( 4 * v39 >= v40 )
            {
              sub_20108A0(a1 + 1288, 2 * v37);
              v35 = v75;
              if ( (*(_BYTE *)(a1 + 1296) & 1) != 0 )
              {
                v45 = a1 + 1304;
                v46 = 7;
              }
              else
              {
                v52 = *(_DWORD *)(a1 + 1312);
                v45 = *(_QWORD *)(a1 + 1304);
                if ( !v52 )
                  goto LABEL_137;
                v46 = v52 - 1;
              }
              v47 = v46 & (37 * v34);
              v32 = (unsigned int *)(v45 + 8LL * v47);
              v20 = (__m128i *)*v32;
              if ( v34 != (_DWORD)v20 )
              {
                v21 = 1;
                v51 = 0;
                while ( (_DWORD)v20 != -1 )
                {
                  if ( !v51 && (_DWORD)v20 == -2 )
                    v51 = v32;
                  v47 = v46 & (v21 + v47);
                  v32 = (unsigned int *)(v45 + 8LL * v47);
                  v20 = (__m128i *)*v32;
                  if ( v34 == (_DWORD)v20 )
                    goto LABEL_62;
                  v21 = (unsigned int)(v21 + 1);
                }
LABEL_68:
                if ( v51 )
                  v32 = v51;
              }
            }
            else
            {
              v41 = v37 - *(_DWORD *)(a1 + 1300) - v39;
              v20 = (__m128i *)(v37 >> 3);
              if ( v41 > (unsigned int)v20 )
              {
LABEL_38:
                *(_DWORD *)(a1 + 1296) = (2 * (v38 >> 1) + 2) | v38 & 1;
                if ( *v32 != -1 )
                  --*(_DWORD *)(a1 + 1300);
                *v32 = v34;
                v32[1] = 0;
                goto LABEL_24;
              }
              sub_20108A0(a1 + 1288, v37);
              v35 = v75;
              if ( (*(_BYTE *)(a1 + 1296) & 1) != 0 )
              {
                v48 = a1 + 1304;
                v49 = 7;
              }
              else
              {
                v53 = *(_DWORD *)(a1 + 1312);
                v48 = *(_QWORD *)(a1 + 1304);
                if ( !v53 )
                {
LABEL_137:
                  *(_DWORD *)(a1 + 1296) = (2 * (*(_DWORD *)(a1 + 1296) >> 1) + 2) | *(_DWORD *)(a1 + 1296) & 1;
                  BUG();
                }
                v49 = v53 - 1;
              }
              v50 = v49 & (37 * v34);
              v32 = (unsigned int *)(v48 + 8LL * v50);
              v20 = (__m128i *)*v32;
              if ( v34 != (_DWORD)v20 )
              {
                v21 = 1;
                v51 = 0;
                while ( (_DWORD)v20 != -1 )
                {
                  if ( !v51 && (_DWORD)v20 == -2 )
                    v51 = v32;
                  v50 = v49 & (v21 + v50);
                  v32 = (unsigned int *)(v48 + 8LL * v50);
                  v20 = (__m128i *)*v32;
                  if ( v34 == (_DWORD)v20 )
                    goto LABEL_62;
                  v21 = (unsigned int)(v21 + 1);
                }
                goto LABEL_68;
              }
            }
LABEL_62:
            v38 = *(_DWORD *)(a1 + 1296);
            goto LABEL_38;
          }
        }
        goto LABEL_12;
      }
    }
  }
  while ( (unsigned __int8)sub_1D18C40(a2, a3) );
  v42 = v91;
  *(_QWORD *)(v84 + 664) = v83;
  if ( v42 != v93 )
    _libc_free((unsigned __int64)v42);
  if ( (v88 & 1) == 0 )
    j___libc_free_0(v89);
}
