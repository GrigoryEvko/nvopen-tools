// Function: sub_1D6A950
// Address: 0x1d6a950
//
__int64 __fastcall sub_1D6A950(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  _QWORD *v13; // rdx
  __int64 *v15; // rdi
  __int64 *v16; // r9
  unsigned int v17; // eax
  char v18; // dl
  __int64 v19; // r13
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // r15
  __int64 *v23; // rsi
  __int64 *v24; // rax
  __int64 *v25; // rcx
  __int64 v26; // rbx
  unsigned int i; // r14d
  _QWORD *v28; // rax
  int v29; // r8d
  int v30; // r9d
  unsigned int v31; // esi
  __int64 v32; // rdi
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // rcx
  char v36; // al
  _QWORD *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  int v40; // edx
  unsigned int v41; // eax
  __int64 *v42; // rcx
  __int64 v43; // rsi
  unsigned int v44; // eax
  _QWORD *v45; // rdi
  __int64 v46; // rsi
  _QWORD *v47; // rax
  int v48; // r9d
  int v49; // edx
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // r12
  int v53; // r8d
  __int64 *v54; // rax
  int v55; // r10d
  unsigned int v56; // ecx
  __int64 *v57; // rdx
  __int64 v58; // r11
  int v60; // edx
  int v61; // ebx
  __int64 v62; // rdx
  __int64 v63; // rcx
  _QWORD *v64; // rdx
  int v65; // r11d
  __int64 *v66; // r10
  int v67; // ebx
  int v68; // ecx
  __int64 v69; // rdi
  int v70; // ecx
  int v71; // r9d
  _QWORD *v73; // [rsp+10h] [rbp-2A0h]
  __int64 *v74; // [rsp+28h] [rbp-288h] BYREF
  __int64 v75; // [rsp+30h] [rbp-280h] BYREF
  __int64 v76; // [rsp+38h] [rbp-278h]
  _QWORD *v77; // [rsp+40h] [rbp-270h] BYREF
  __int64 v78; // [rsp+48h] [rbp-268h]
  _QWORD v79[32]; // [rsp+50h] [rbp-260h] BYREF
  __int64 v80; // [rsp+150h] [rbp-160h] BYREF
  __int64 *v81; // [rsp+158h] [rbp-158h]
  __int64 *v82; // [rsp+160h] [rbp-150h]
  __int64 v83; // [rsp+168h] [rbp-148h]
  int v84; // [rsp+170h] [rbp-140h]
  _BYTE v85[312]; // [rsp+178h] [rbp-138h] BYREF

  v13 = v79;
  v15 = (__int64 *)v85;
  v16 = (__int64 *)v85;
  v78 = 0x2000000001LL;
  v17 = 1;
  v77 = v79;
  v80 = 0;
  v81 = (__int64 *)v85;
  v82 = (__int64 *)v85;
  v83 = 32;
  v84 = 0;
  v79[0] = a2;
  while ( 1 )
  {
    v22 = v13[v17 - 1];
    LODWORD(v78) = v17 - 1;
    if ( v15 != v16 )
      goto LABEL_2;
    v23 = &v15[HIDWORD(v83)];
    if ( v23 == v15 )
    {
LABEL_51:
      if ( HIDWORD(v83) >= (unsigned int)v83 )
      {
LABEL_2:
        sub_16CCBA0((__int64)&v80, v22);
        v16 = v82;
        v15 = v81;
        if ( !v18 )
          goto LABEL_6;
      }
      else
      {
        ++HIDWORD(v83);
        *v23 = v22;
        v15 = v81;
        ++v80;
        v16 = v82;
      }
LABEL_3:
      if ( *(_BYTE *)(v22 + 16) <= 0x17u )
        goto LABEL_6;
      v19 = sub_13E3350(v22, *(const __m128i **)(a1 + 32), 0, 1, a13);
      if ( !v19 )
      {
LABEL_5:
        v16 = v82;
        v15 = v81;
        goto LABEL_6;
      }
      v26 = *(_QWORD *)(v22 + 8);
      for ( i = v78; v26; v26 = *(_QWORD *)(v26 + 8) )
      {
        v28 = sub_1648700(v26);
        if ( HIDWORD(v78) <= i )
        {
          v73 = v28;
          sub_16CD150((__int64)&v77, v79, 0, 8, v29, v30);
          i = v78;
          v28 = v73;
        }
        v77[i] = v28;
        i = v78 + 1;
        LODWORD(v78) = v78 + 1;
      }
      v31 = *(_DWORD *)(a1 + 24);
      v75 = v22;
      v76 = v19;
      if ( v31 )
      {
        v32 = *(_QWORD *)(a1 + 8);
        v33 = (v31 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v34 = (__int64 *)(v32 + 16LL * v33);
        v35 = *v34;
        if ( v22 == *v34 )
          goto LABEL_23;
        v65 = 1;
        v66 = 0;
        while ( v35 != -8 )
        {
          if ( v35 == -16 && !v66 )
            v66 = v34;
          v33 = (v31 - 1) & (v65 + v33);
          v34 = (__int64 *)(v32 + 16LL * v33);
          v35 = *v34;
          if ( v22 == *v34 )
            goto LABEL_23;
          ++v65;
        }
        v67 = *(_DWORD *)(a1 + 16);
        if ( v66 )
          v34 = v66;
        ++*(_QWORD *)a1;
        v68 = v67 + 1;
        if ( 4 * (v67 + 1) < 3 * v31 )
        {
          v69 = v22;
          if ( v31 - *(_DWORD *)(a1 + 20) - v68 > v31 >> 3 )
          {
LABEL_69:
            *(_DWORD *)(a1 + 16) = v68;
            if ( *v34 != -8 )
              --*(_DWORD *)(a1 + 20);
            *v34 = v69;
            v34[1] = v76;
LABEL_23:
            sub_164D160(v22, v19, a3, a4, a5, a6, v20, v21, a9, a10);
            v36 = *(_BYTE *)(v22 + 16);
            if ( v36 == 77 )
            {
              v75 = v22;
              if ( (*(_BYTE *)(a1 + 48) & 1) != 0 )
              {
                v39 = a1 + 56;
                v40 = 31;
              }
              else
              {
                v49 = *(_DWORD *)(a1 + 64);
                v39 = *(_QWORD *)(a1 + 56);
                if ( !v49 )
                  goto LABEL_28;
                v40 = v49 - 1;
              }
              v41 = v40 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
              v42 = (__int64 *)(v39 + 8LL * v41);
              v43 = *v42;
              if ( v22 != *v42 )
              {
                v70 = 1;
                while ( v43 != -8 )
                {
                  v71 = v70 + 1;
                  v41 = v40 & (v70 + v41);
                  v42 = (__int64 *)(v39 + 8LL * v41);
                  v43 = *v42;
                  if ( v22 == *v42 )
                    goto LABEL_32;
                  v70 = v71;
                }
                goto LABEL_28;
              }
LABEL_32:
              *v42 = -16;
              v44 = *(_DWORD *)(a1 + 48);
              ++*(_DWORD *)(a1 + 52);
              v45 = *(_QWORD **)(a1 + 312);
              *(_DWORD *)(a1 + 48) = (2 * (v44 >> 1) - 2) | v44 & 1;
              v46 = (__int64)&v45[*(unsigned int *)(a1 + 320)];
              v47 = sub_1D5A7E0(v45, v46, &v75);
              if ( v47 + 1 != (_QWORD *)v46 )
              {
                memmove(v47, v47 + 1, v46 - (_QWORD)(v47 + 1));
                v48 = *(_DWORD *)(a1 + 320);
              }
              *(_DWORD *)(a1 + 320) = v48 - 1;
              v36 = *(_BYTE *)(v22 + 16);
            }
            if ( v36 == 79 )
            {
              v37 = *(_QWORD **)(a1 + 592);
              if ( *(_QWORD **)(a1 + 600) == v37 )
              {
                v64 = &v37[*(unsigned int *)(a1 + 612)];
                if ( v37 == v64 )
                {
LABEL_61:
                  v37 = v64;
                }
                else
                {
                  while ( v22 != *v37 )
                  {
                    if ( v64 == ++v37 )
                      goto LABEL_61;
                  }
                }
              }
              else
              {
                v37 = sub_16CC9F0(a1 + 584, v22);
                if ( v22 == *v37 )
                {
                  v62 = *(_QWORD *)(a1 + 600);
                  if ( v62 == *(_QWORD *)(a1 + 592) )
                    v63 = *(unsigned int *)(a1 + 612);
                  else
                    v63 = *(unsigned int *)(a1 + 608);
                  v64 = (_QWORD *)(v62 + 8 * v63);
                }
                else
                {
                  v38 = *(_QWORD *)(a1 + 600);
                  if ( v38 != *(_QWORD *)(a1 + 592) )
                    goto LABEL_28;
                  v37 = (_QWORD *)(v38 + 8LL * *(unsigned int *)(a1 + 612));
                  v64 = v37;
                }
              }
              if ( v64 != v37 )
              {
                *v37 = -2;
                ++*(_DWORD *)(a1 + 616);
                sub_15F20C0((_QWORD *)v22);
                goto LABEL_5;
              }
            }
LABEL_28:
            sub_15F20C0((_QWORD *)v22);
            goto LABEL_5;
          }
LABEL_74:
          sub_176F940(a1, v31);
          sub_176A9A0(a1, &v75, &v74);
          v34 = v74;
          v69 = v75;
          v68 = *(_DWORD *)(a1 + 16) + 1;
          goto LABEL_69;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
      }
      v31 *= 2;
      goto LABEL_74;
    }
    v24 = v15;
    v25 = 0;
    while ( v22 != *v24 )
    {
      if ( *v24 == -2 )
        v25 = v24;
      if ( v23 == ++v24 )
      {
        if ( !v25 )
          goto LABEL_51;
        *v25 = v22;
        v16 = v82;
        --v84;
        v15 = v81;
        ++v80;
        goto LABEL_3;
      }
    }
LABEL_6:
    v17 = v78;
    if ( !(_DWORD)v78 )
      break;
    v13 = v77;
  }
  v50 = *(unsigned int *)(a1 + 24);
  v51 = *(_QWORD *)(a1 + 8);
  v52 = a2;
  v53 = v50;
  v54 = (__int64 *)(v51 + 16 * v50);
  v55 = v53 - 1;
  while ( v53 )
  {
    v56 = v55 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
    v57 = (__int64 *)(v51 + 16LL * v56);
    v58 = *v57;
    if ( v52 != *v57 )
    {
      v60 = 1;
      while ( v58 != -8 )
      {
        v61 = v60 + 1;
        v56 = v55 & (v60 + v56);
        v57 = (__int64 *)(v51 + 16LL * v56);
        v58 = *v57;
        if ( v52 == *v57 )
          goto LABEL_39;
        v60 = v61;
      }
      break;
    }
LABEL_39:
    if ( v54 == v57 )
      break;
    v52 = v57[1];
  }
  if ( v16 != v15 )
    _libc_free((unsigned __int64)v16);
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
  return v52;
}
