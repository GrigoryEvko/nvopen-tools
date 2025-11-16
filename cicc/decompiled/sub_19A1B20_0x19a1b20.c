// Function: sub_19A1B20
// Address: 0x19a1b20
//
void __fastcall sub_19A1B20(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        char a9)
{
  __int64 v12; // r14
  char v13; // al
  _QWORD *v14; // r8
  __int64 v15; // rcx
  char *v16; // r12
  __int64 v17; // r8
  char *v18; // r14
  char *v19; // r8
  int v20; // edx
  __int64 *v21; // rdi
  size_t v22; // r9
  unsigned __int64 v23; // r14
  __int64 v24; // rdx
  char *v25; // rax
  size_t v26; // r8
  unsigned __int64 v27; // r14
  __int64 *v28; // r14
  __int64 v29; // rdx
  int v30; // r8d
  int v31; // r9d
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rsi
  int v35; // r8d
  int v36; // r9d
  __int64 v37; // r11
  __int64 v38; // rcx
  unsigned int v39; // eax
  unsigned int v40; // edx
  unsigned int v41; // r8d
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  char v48; // al
  __int64 v49; // r12
  __int64 v50; // rax
  __int64 v51; // rax
  char v52; // al
  __int64 v53; // r8
  __int64 v54; // rcx
  __int64 *v55; // rax
  char v56; // al
  unsigned int v57; // [rsp+10h] [rbp-1E0h]
  char *v59; // [rsp+18h] [rbp-1D8h]
  size_t v60; // [rsp+18h] [rbp-1D8h]
  __int64 v61; // [rsp+18h] [rbp-1D8h]
  __int64 v62; // [rsp+20h] [rbp-1D0h]
  char *v63; // [rsp+28h] [rbp-1C8h]
  char *v64; // [rsp+28h] [rbp-1C8h]
  __int64 v65; // [rsp+28h] [rbp-1C8h]
  __int64 v67; // [rsp+38h] [rbp-1B8h]
  __int64 v68; // [rsp+38h] [rbp-1B8h]
  char *v69; // [rsp+40h] [rbp-1B0h]
  __int64 v70; // [rsp+40h] [rbp-1B0h]
  int v71; // [rsp+40h] [rbp-1B0h]
  __int64 v73; // [rsp+58h] [rbp-198h] BYREF
  void *src; // [rsp+60h] [rbp-190h] BYREF
  __int64 v75; // [rsp+68h] [rbp-188h]
  _BYTE v76[64]; // [rsp+70h] [rbp-180h] BYREF
  __int64 *v77; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v78; // [rsp+B8h] [rbp-138h]
  _BYTE v79[64]; // [rsp+C0h] [rbp-130h] BYREF
  _QWORD v80[2]; // [rsp+100h] [rbp-F0h] BYREF
  char v81; // [rsp+110h] [rbp-E0h]
  __int64 v82; // [rsp+118h] [rbp-D8h]
  _BYTE *v83; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v84; // [rsp+128h] [rbp-C8h]
  _BYTE v85[32]; // [rsp+130h] [rbp-C0h] BYREF
  __int64 *v86; // [rsp+150h] [rbp-A0h]
  __int64 v87; // [rsp+158h] [rbp-98h]
  _QWORD v88[2]; // [rsp+160h] [rbp-90h] BYREF
  char v89; // [rsp+170h] [rbp-80h]
  __int64 v90; // [rsp+178h] [rbp-78h]
  unsigned __int64 v91[2]; // [rsp+180h] [rbp-70h] BYREF
  _BYTE v92[32]; // [rsp+190h] [rbp-60h] BYREF
  __int64 v93; // [rsp+1B0h] [rbp-40h]
  __int64 v94; // [rsp+1B8h] [rbp-38h]

  if ( a9 )
    v12 = *(_QWORD *)(a4 + 80);
  else
    v12 = *(_QWORD *)(*(_QWORD *)(a4 + 32) + 8 * a6);
  v13 = sub_14A2B00(a1[4]);
  v14 = (_QWORD *)a1[1];
  v15 = a1[5];
  if ( !v13 || *(_DWORD *)(a2 + 32) != 2 )
    goto LABEL_5;
  v47 = *(_QWORD *)(a2 + 40);
  v48 = *(_BYTE *)(v47 + 8);
  if ( v48 == 16 )
    v48 = *(_BYTE *)(**(_QWORD **)(v47 + 16) + 8LL);
  if ( v48 != 11 || *(_WORD *)(v12 + 24) != 7 )
  {
LABEL_5:
    src = v76;
    v75 = 0x800000000LL;
    v73 = sub_199A990(v12, 0, (__int64)&src, v15, v14, 0, a7, a8);
    if ( v73 )
      sub_1458920((__int64)&src, &v73);
    v16 = (char *)src;
    if ( (unsigned int)v75 == 1 || (v69 = (char *)src + 8 * (unsigned int)v75, v69 == src) )
    {
LABEL_40:
      if ( v16 != v76 )
        _libc_free((unsigned __int64)v16);
      return;
    }
    while ( 1 )
    {
      v17 = *(_QWORD *)v16;
      v18 = v16;
      v16 += 8;
      if ( *(_WORD *)(v17 + 24) == 10 )
      {
        if ( !sub_146CEE0(a1[1], v17, a1[5]) )
          goto LABEL_38;
        v17 = *((_QWORD *)v16 - 1);
      }
      if ( !sub_199E790(
              (__int64 *)a1[4],
              a1[1],
              *(_QWORD *)(a2 + 712),
              *(_QWORD *)(a2 + 720),
              *(_DWORD *)(a2 + 32),
              *(_QWORD *)(a2 + 40),
              a7,
              a8,
              *(_DWORD *)(a2 + 48),
              v17,
              *(unsigned int *)(a4 + 40) - ((unsigned __int64)(*(_QWORD *)(a4 + 80) == 0) - 1) > 1) )
      {
        v19 = (char *)src;
        v20 = 0;
        v21 = (__int64 *)v79;
        v22 = v18 - (_BYTE *)src;
        v77 = (__int64 *)v79;
        v78 = 0x800000000LL;
        v23 = (v18 - (_BYTE *)src) >> 3;
        if ( v22 > 0x40 )
        {
          v60 = v22;
          v64 = (char *)src;
          sub_16CD150((__int64)&v77, v79, v23, 8, (int)src, v22);
          v20 = v78;
          v22 = v60;
          v19 = v64;
          v21 = &v77[(unsigned int)v78];
        }
        if ( v19 != v16 - 8 )
        {
          memcpy(v21, v19, v22);
          v20 = v78;
        }
        LODWORD(v78) = v23 + v20;
        v24 = (unsigned int)(v23 + v20);
        v25 = (char *)src + 8 * (unsigned int)v75;
        v26 = v25 - v16;
        v27 = (v25 - v16) >> 3;
        if ( v27 > (unsigned __int64)HIDWORD(v78) - v24 )
        {
          v59 = (char *)((_BYTE *)src + 8 * (unsigned int)v75 - v16);
          v63 = (char *)src + 8 * (unsigned int)v75;
          sub_16CD150((__int64)&v77, v79, v27 + v24, 8, v26, v22);
          v24 = (unsigned int)v78;
          v26 = (size_t)v59;
          v25 = v63;
        }
        if ( v25 != v16 )
        {
          memcpy(&v77[v24], v16, v26);
          LODWORD(v24) = v78;
        }
        LODWORD(v78) = v27 + v24;
        if ( (_DWORD)v27 + (_DWORD)v24 != 1
          || !sub_199E790(
                (__int64 *)a1[4],
                a1[1],
                *(_QWORD *)(a2 + 712),
                *(_QWORD *)(a2 + 720),
                *(_DWORD *)(a2 + 32),
                *(_QWORD *)(a2 + 40),
                a7,
                a8,
                *(_DWORD *)(a2 + 48),
                *v77,
                *(unsigned int *)(a4 + 40) - ((unsigned __int64)(*(_QWORD *)(a4 + 80) == 0) - 1) > 1) )
        {
          v28 = sub_147DD40(a1[1], (__int64 *)&v77, 0, 0, a7, a8);
          if ( !sub_14560B0((__int64)v28) )
          {
            v32 = *(unsigned int *)(a4 + 40);
            v80[0] = *(_QWORD *)a4;
            v80[1] = *(_QWORD *)(a4 + 8);
            v81 = *(_BYTE *)(a4 + 16);
            v82 = *(_QWORD *)(a4 + 24);
            v83 = v85;
            v84 = 0x400000000LL;
            if ( (_DWORD)v32 )
              sub_19930D0((__int64)&v83, a4 + 32, v29, v32, v30, v31);
            v86 = *(__int64 **)(a4 + 80);
            v87 = *(_QWORD *)(a4 + 88);
            if ( a9 )
            {
              v86 = v28;
              v33 = (unsigned int)v84;
              if ( (unsigned int)v84 < HIDWORD(v84) )
                goto LABEL_25;
LABEL_48:
              sub_16CD150((__int64)&v83, v85, 0, 8, v30, v31);
              v33 = (unsigned int)v84;
            }
            else
            {
              *(_QWORD *)&v83[8 * a6] = v28;
              v33 = (unsigned int)v84;
              if ( (unsigned int)v84 >= HIDWORD(v84) )
                goto LABEL_48;
            }
LABEL_25:
            *(_QWORD *)&v83[8 * v33] = *((_QWORD *)v16 - 1);
            v34 = a1[5];
            LODWORD(v84) = v84 + 1;
            sub_19932F0((__int64)v80, v34);
            if ( (unsigned __int8)sub_19A1660((__int64)a1, a2, a3, v37, v35, v36) )
            {
              v39 = 0x3FFFFFFF;
              if ( (_DWORD)v75 )
              {
                _BitScanReverse(&v40, v75);
                v39 = (31 - (v40 ^ 0x1F)) >> 2;
              }
              v41 = v39 + a5 + 1;
              v42 = *(_QWORD *)(a2 + 744) + 96LL * *(unsigned int *)(a2 + 752) - 96;
              v88[0] = *(_QWORD *)v42;
              v88[1] = *(_QWORD *)(v42 + 8);
              v89 = *(_BYTE *)(v42 + 16);
              v43 = *(_QWORD *)(v42 + 24);
              v91[1] = 0x400000000LL;
              v90 = v43;
              v44 = *(unsigned int *)(v42 + 40);
              v91[0] = (unsigned __int64)v92;
              if ( (_DWORD)v44 )
              {
                v57 = v41;
                v61 = v42;
                sub_19930D0((__int64)v91, v42 + 32, v44, v38, v41, (int)v92);
                v41 = v57;
                v42 = v61;
              }
              v45 = *(_QWORD *)(v42 + 80);
              v46 = *(_QWORD *)(v42 + 88);
              v93 = v45;
              v94 = v46;
              if ( v41 <= 2 )
                sub_19A22F0(a1, a2, a3, v88);
              if ( (_BYTE *)v91[0] != v92 )
                _libc_free(v91[0]);
            }
            if ( v83 != v85 )
              _libc_free((unsigned __int64)v83);
          }
        }
        if ( v77 != (__int64 *)v79 )
          _libc_free((unsigned __int64)v77);
      }
LABEL_38:
      if ( v69 == v16 )
      {
        v16 = (char *)src;
        goto LABEL_40;
      }
    }
  }
  v49 = a1[4];
  v67 = a1[5];
  v70 = a1[1];
  v50 = sub_13A5BC0((_QWORD *)v12, v70);
  if ( *(_WORD *)(v50 + 24) )
    goto LABEL_55;
  v65 = v67;
  v68 = v70;
  v62 = v50;
  v71 = sub_16431D0(*(_QWORD *)(a2 + 40));
  v51 = sub_1456040(v62);
  if ( v71 != (unsigned int)sub_16431D0(v51) )
    goto LABEL_55;
  sub_1456040(**(_QWORD **)(v12 + 32));
  v52 = sub_14A3850(v49);
  v53 = v68;
  v54 = v65;
  if ( !v52 )
  {
    sub_1456040(**(_QWORD **)(v12 + 32));
    v56 = sub_14A3880(v49);
    v53 = v68;
    v54 = v65;
    if ( !v56 )
      goto LABEL_55;
  }
  v55 = *(__int64 **)(v12 + 32);
  if ( !*(_WORD *)(*v55 + 24) || !sub_146CEE0(v53, *v55, v54) )
  {
LABEL_55:
    v14 = (_QWORD *)a1[1];
    v15 = a1[5];
    goto LABEL_5;
  }
}
