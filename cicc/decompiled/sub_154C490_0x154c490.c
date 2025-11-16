// Function: sub_154C490
// Address: 0x154c490
//
void __fastcall sub_154C490(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7)
{
  char *v7; // r13
  unsigned int v10; // r14d
  int v11; // r12d
  int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  char **v20; // rax
  _BYTE *v21; // rdi
  char v22; // al
  int v23; // eax
  __int64 v24; // rcx
  int v25; // eax
  __int64 v26; // rsi
  unsigned int v27; // r10d
  __int64 *v28; // rdx
  __int64 v29; // r12
  __int64 v30; // rcx
  __int64 *v31; // r12
  __int64 *v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 *v37; // r14
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdi
  __int64 v43; // r9
  _BYTE *v44; // rsi
  _BYTE *v45; // rax
  unsigned int v46; // edx
  unsigned int v47; // ecx
  _QWORD *v48; // r12
  __int64 v49; // rax
  __int64 v50; // rdx
  size_t v51; // r14
  char *v52; // rax
  __int64 v53; // rax
  int v54; // eax
  int v55; // r10d
  int v56; // edx
  int v57; // r11d
  __int128 v58; // [rsp-20h] [rbp-510h]
  __int64 v59; // [rsp+8h] [rbp-4E8h]
  int v60; // [rsp+2Ch] [rbp-4C4h] BYREF
  __int64 v61; // [rsp+30h] [rbp-4C0h] BYREF
  __int64 v62; // [rsp+38h] [rbp-4B8h] BYREF
  bool v63; // [rsp+47h] [rbp-4A9h] BYREF
  unsigned __int64 v64[5]; // [rsp+48h] [rbp-4A8h] BYREF
  __int64 v65; // [rsp+70h] [rbp-480h]
  bool *v66; // [rsp+78h] [rbp-478h]
  int *v67; // [rsp+80h] [rbp-470h]
  __int64 v68; // [rsp+90h] [rbp-460h]
  bool *v69; // [rsp+98h] [rbp-458h]
  int *v70; // [rsp+A0h] [rbp-450h]
  _BYTE *v71; // [rsp+B0h] [rbp-440h] BYREF
  __int64 v72; // [rsp+B8h] [rbp-438h]
  _BYTE v73[1072]; // [rsp+C0h] [rbp-430h] BYREF

  v61 = a2;
  v71 = v73;
  v72 = 0x4000000000LL;
  v7 = *(char **)(a1 + 8);
  v62 = a1;
  v60 = a3;
  if ( v7 )
  {
    v10 = 0;
    do
    {
      v11 = *(_DWORD *)(a4 + 24);
      if ( v11 )
      {
        v12 = v11 - 1;
        v13 = sub_1648700(v7);
        v14 = *(_QWORD *)(a4 + 8);
        v15 = v13;
        v16 = v12 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v17 = (__int64 *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( v15 == *v17 )
        {
LABEL_5:
          if ( *((_DWORD *)v17 + 2) )
          {
            v19 = v10;
            if ( HIDWORD(v72) <= v10 )
            {
              sub_16CD150(&v71, v73, 0, 16);
              v19 = (unsigned int)v72;
            }
            v20 = (char **)&v71[16 * v19];
            *v20 = v7;
            v20[1] = (char *)v10;
            v10 = v72 + 1;
            LODWORD(v72) = v72 + 1;
          }
        }
        else
        {
          v54 = 1;
          while ( v18 != -8 )
          {
            v55 = v54 + 1;
            v16 = v12 & (v54 + v16);
            v17 = (__int64 *)(v14 + 16LL * v16);
            v18 = *v17;
            if ( v15 == *v17 )
              goto LABEL_5;
            v54 = v55;
          }
        }
      }
      v7 = (char *)*((_QWORD *)v7 + 1);
    }
    while ( v7 );
    v21 = v71;
    if ( v10 > 1 )
    {
      v22 = *(_BYTE *)(v62 + 16);
      if ( v22 == 3 )
      {
        v63 = 0;
      }
      else
      {
        v63 = v22 != 0 && v22 != 18;
        if ( v22 == 4 )
        {
          v23 = *(_DWORD *)(a4 + 24);
          if ( v23 )
          {
            v24 = *(_QWORD *)(v62 - 24);
            v25 = v23 - 1;
            v26 = *(_QWORD *)(a4 + 8);
            v27 = v25 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v28 = (__int64 *)(v26 + 16LL * v27);
            v29 = *v28;
            if ( v24 == *v28 )
            {
LABEL_15:
              v23 = *((_DWORD *)v28 + 2);
            }
            else
            {
              v56 = 1;
              while ( v29 != -8 )
              {
                v57 = v56 + 1;
                v27 = v25 & (v56 + v27);
                v28 = (__int64 *)(v26 + 16LL * v27);
                v29 = *v28;
                if ( v24 == *v28 )
                  goto LABEL_15;
                v56 = v57;
              }
              v23 = 0;
            }
          }
          v60 = v23;
        }
      }
      v64[3] = (unsigned __int64)&v60;
      v30 = 16LL * v10;
      v31 = (__int64 *)&v71[v30];
      v32 = (__int64 *)&v71[v30];
      _BitScanReverse64((unsigned __int64 *)&v30, v30 >> 4);
      v64[2] = (unsigned __int64)&v63;
      *((_QWORD *)&v58 + 1) = &v63;
      *(_QWORD *)&v58 = a4;
      v59 = (__int64)v71;
      v64[1] = a4;
      sub_1549160((__int64)v71, v32, 2LL * (int)(63 - (v30 ^ 0x3F)), v30 ^ 0x3F, a5, a6, v58, (__int64)&v60);
      if ( 16 * (unsigned __int64)v10 <= 0x100 )
      {
        v65 = a4;
        v66 = &v63;
        v67 = &v60;
        sub_1548EC0(v59, v31, v33, v34, v35, v36, a7);
      }
      else
      {
        v37 = (__int64 *)(v59 + 256);
        v65 = a4;
        v67 = &v60;
        v66 = &v63;
        sub_1548EC0(v59, (__int64 *)(v59 + 256), v33, v34, v35, v36, a7);
        if ( v31 != (__int64 *)(v59 + 256) )
        {
          do
          {
            v42 = (__int64)v37;
            v37 += 2;
            v68 = a4;
            v69 = &v63;
            v70 = &v60;
            sub_1548E60(v42, v59 + 256, v38, v39, v40, v41, a4);
          }
          while ( v31 != v37 );
        }
      }
      v43 = (unsigned int)v72;
      v21 = v71;
      v44 = &v71[16 * (unsigned int)v72];
      if ( v44 != v71 )
      {
        v45 = v71 + 16;
        if ( v44 != v71 + 16 )
        {
          v46 = *((_DWORD *)v71 + 2);
          while ( 1 )
          {
            v47 = v46;
            v46 = *((_DWORD *)v45 + 2);
            if ( v46 < v47 )
              break;
            v45 += 16;
            if ( v44 == v45 )
              goto LABEL_36;
          }
          if ( v44 != v45 )
          {
            v64[0] = (unsigned int)v72;
            v48 = *(_QWORD **)(a5 + 8);
            if ( v48 == *(_QWORD **)(a5 + 16) )
            {
              sub_153EFA0((__int64 *)a5, *(char **)(a5 + 8), &v62, &v61, v64);
              v43 = (unsigned int)v72;
              v21 = v71;
            }
            else
            {
              if ( v48 )
              {
                v49 = v61;
                v50 = v62;
                v48[2] = 0;
                v51 = 4 * v43;
                v48[3] = 0;
                *v48 = v50;
                v48[1] = v49;
                v48[4] = 0;
                if ( v43 )
                {
                  v52 = (char *)sub_22077B0(4 * v43);
                  v7 = &v52[v51];
                  v48[2] = v52;
                  v48[4] = &v52[v51];
                  if ( v52 != &v52[v51] )
                    memset(v52, 0, v51);
                }
                v48[3] = v7;
                v43 = (unsigned int)v72;
                v48 = *(_QWORD **)(a5 + 8);
                v21 = v71;
              }
              *(_QWORD *)(a5 + 8) = v48 + 5;
            }
            if ( v43 )
            {
              v53 = 0;
              do
              {
                *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a5 + 8) - 24LL) + v53) = *(_DWORD *)&v21[4 * v53 + 8];
                v53 += 4;
                v21 = v71;
              }
              while ( 4 * v43 != v53 );
            }
          }
        }
      }
    }
LABEL_36:
    if ( v21 != v73 )
      _libc_free((unsigned __int64)v21);
  }
}
